"""OpenRouter API backend for moral summarization inference.

Drop-in replacement for ``LlamaModelForSequenceCompletion`` that calls the
OpenRouter chat-completions API (https://openrouter.ai) instead of loading a
model locally. Lets you run large hosted models (e.g.
``meta-llama/llama-3.3-70b-instruct``) without any GPU.

Exposes the same ``get_response(query, system_content, ...)`` signature used by
prompting.py, returning ``(response, conversation)`` where ``response`` is a
list of message dicts (so callers can use ``response[-1]['content']``).

Config (under an ``openrouter`` key) + the API key:
    backend: openrouter
    openrouter:
      model: meta-llama/llama-3.3-70b-instruct
      # api_key can be set here, but prefer the OPENROUTER_API_KEY env var.
      request_timeout: 120
      max_retries: 4

The API key is read from (in order): config['openrouter']['api_key'], then the
OPENROUTER_API_KEY environment variable.
"""
import os
import time

import requests


class ModerationRefusal(Exception):
    """Raised when the provider's input moderation flags the prompt.

    This is deterministic (the same input will always be flagged), so callers
    should record it as a refusal and move on rather than retry.
    """
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class OpenRouterModelForSequenceCompletion:
    def __init__(self, config):
        self.config = config
        self.verbose = config.get('verbose', True)

        orc = config.get('openrouter', {})
        self.model_name = orc.get('model', 'meta-llama/llama-3.3-70b-instruct:free')
        self.base_url = orc.get('base_url', 'https://openrouter.ai/api/v1')
        self.chat_url = f"{self.base_url.rstrip('/')}/chat/completions"
        self.request_timeout = orc.get('request_timeout', 120)
        # Some providers (e.g. JAX-based gpt-oss on OpenInference) reject a
        # per-request `seed` and fail every call. Set use_seed: false to omit it.
        self.use_seed = orc.get('use_seed', True)
        # Free-tier models are heavily rate-limited upstream, so default to many
        # retries with long-ish waits. The API's Retry-After header is honored
        # when present. Tune via config: max_retries, max_backoff.
        self.max_retries = orc.get('max_retries', 30)
        self.max_backoff = orc.get('max_backoff', 60)  # cap per-wait seconds

        self.api_key = self._resolve_api_key(orc)
        if not self.api_key:
            raise RuntimeError(
                "No OpenRouter API key found. Provide it via (in order): "
                "config['openrouter']['api_key'], OPENROUTER_API_KEY env var, or "
                "an `api_key_file` path in the config (default: models/openrouter_api_key)."
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.verbose:
            print(f"Using OpenRouter backend: model={self.model_name}")

    @staticmethod
    def _resolve_api_key(orc):
        """Find the key from (in order): config api_key, env var, key file."""
        if orc.get('api_key'):
            return orc['api_key'].strip()
        if os.environ.get('OPENROUTER_API_KEY'):
            return os.environ['OPENROUTER_API_KEY'].strip()
        key_file = orc.get('api_key_file', 'models/openrouter_api_key')
        if key_file and os.path.isfile(key_file):
            with open(key_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return None

    def get_response(
            self,
            query,
            system_content="You are a news summarizer assistant and a moral expert.",
            max_tokens=4096,
            temperature=0.6,
            top_p=0.9,
            ):
        conversation = [{"role": "system", "content": system_content}]
        prompt = conversation + [{"role": "user", "content": query}]

        payload = {
            "model": self.model_name,
            "messages": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if self.use_seed:
            payload["seed"] = 0

        if self.verbose:
            print("Prompting the model...")
        start_time = time.time()

        data = self._post_with_retries(payload)

        if self.verbose:
            print(f"Execution time: {time.time() - start_time:.1f}s")

        content = data["choices"][0]["message"]["content"]
        assistant_message = {"role": "assistant", "content": content}
        response = [assistant_message]
        return response, prompt + [assistant_message]

    def _post_with_retries(self, payload):
        """POST with retries on transient errors (429/403/5xx/timeouts).

        Free-tier models throttle aggressively. We honor the API's Retry-After
        signal when present, otherwise use exponential backoff capped at
        ``max_backoff``. 403 is included because OpenRouter sometimes returns it
        as a transient variant of upstream rate-limiting.
        """
        last_err = None
        for attempt in range(self.max_retries):
            retry_after = None
            try:
                resp = requests.post(
                    self.chat_url, headers=self.headers, json=payload,
                    timeout=self.request_timeout,
                )
                if resp.status_code == 200:
                    body = resp.json()
                    # OpenRouter can return 200 with an {"error": ...} body and
                    # no 'choices' (provider hiccup, transient upstream error).
                    # Treat that as retryable instead of crashing downstream.
                    if 'choices' in body and body['choices']:
                        return body
                    err = body.get('error', {})
                    last_err = f"200 but no choices: {str(err)[:200] or str(body)[:200]}"
                    retry_after = self._extract_retry_after(resp)
                # A 403 from input moderation is deterministic - don't retry,
                # signal a refusal so the caller can record it and move on.
                elif resp.status_code == 403 and self._is_moderation_block(resp):
                    raise ModerationRefusal(self._moderation_reason(resp))
                # Retry on rate-limit / forbidden(transient) / server errors.
                elif resp.status_code in (429, 403, 500, 502, 503, 504):
                    last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    retry_after = self._extract_retry_after(resp)
                else:
                    # genuine client error (e.g. 400 bad model, 401 bad key)
                    resp.raise_for_status()
            except requests.exceptions.RequestException as e:
                last_err = str(e)

            if attempt < self.max_retries - 1:
                if retry_after is not None:
                    wait = min(retry_after + 1, self.max_backoff)
                else:
                    wait = min(2.0 ** min(attempt, 6), self.max_backoff)
                if self.verbose:
                    print(f"  request failed ({last_err}); retry "
                          f"{attempt + 1}/{self.max_retries} in {wait:.0f}s")
                time.sleep(wait)

        raise RuntimeError(
            f"OpenRouter request failed after {self.max_retries} attempts: {last_err}"
        )

    @staticmethod
    def _is_moderation_block(resp):
        """True if a 403 is due to input content moderation (not a transient block)."""
        text = resp.text.lower()
        return 'moderation' in text or 'flagged' in text

    @staticmethod
    def _moderation_reason(resp):
        """Extract the moderation reason(s) for logging / the refusal record."""
        try:
            meta = resp.json().get('error', {}).get('metadata', {})
            reasons = meta.get('reasons')
            if reasons:
                return ', '.join(reasons)
            msg = resp.json().get('error', {}).get('message')
            if msg:
                return msg
        except (ValueError, AttributeError):
            pass
        return 'input flagged by provider moderation'

    @staticmethod
    def _extract_retry_after(resp):
        """Pull a retry delay (seconds) from the Retry-After header or JSON body."""
        # Standard HTTP header
        hdr = resp.headers.get('Retry-After')
        if hdr:
            try:
                return float(hdr)
            except ValueError:
                pass
        # OpenRouter nests retry_after_seconds in error.metadata
        try:
            meta = resp.json().get('error', {}).get('metadata', {})
            if 'retry_after_seconds' in meta:
                return float(meta['retry_after_seconds'])
        except (ValueError, AttributeError):
            pass
        return None

