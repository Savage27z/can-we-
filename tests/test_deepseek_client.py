import unittest
from unittest.mock import MagicMock, patch

from narration import config, deepseek_client


def fake_response(status_code=200, content="hello", text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


class DeepSeekClientTests(unittest.TestCase):
    def setUp(self):
        config.DEEPSEEK_API_KEY = "test-key"

    def test_chat_returns_message_content(self):
        with patch.object(deepseek_client.requests, "post",
                           return_value=fake_response(content="the report text")) as mock_post:
            result = deepseek_client.chat("system", "user")
        self.assertEqual(result, "the report text")
        # Verify the request shape (model, messages, auth header) is well-formed.
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["model"], config.DEEPSEEK_MODEL)
        self.assertEqual(kwargs["json"]["messages"][0], {"role": "system", "content": "system"})
        self.assertEqual(kwargs["json"]["messages"][1], {"role": "user", "content": "user"})
        self.assertIn("Bearer test-key", kwargs["headers"]["Authorization"])

    def test_raises_on_http_error(self):
        with patch.object(deepseek_client.requests, "post",
                           return_value=fake_response(status_code=401, text="Unauthorized")):
            with self.assertRaises(deepseek_client.DeepSeekAPIError):
                deepseek_client.chat("system", "user")

    def test_missing_api_key_raises_before_request(self):
        config.DEEPSEEK_API_KEY = None
        with patch.object(deepseek_client.requests, "post") as mock_post:
            with self.assertRaises(RuntimeError):
                deepseek_client.chat("system", "user")
        mock_post.assert_not_called()

    def test_raises_typed_error_on_malformed_200_response(self):
        # e.g. content filtering returning an empty "choices" list.
        resp = MagicMock()
        resp.status_code = 200
        resp.text = '{"choices": []}'
        resp.json.return_value = {"choices": []}
        with patch.object(deepseek_client.requests, "post", return_value=resp):
            with self.assertRaises(deepseek_client.DeepSeekAPIError):
                deepseek_client.chat("system", "user")

    def test_raises_typed_error_on_invalid_json(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "not json"
        resp.json.side_effect = ValueError("no JSON object could be decoded")
        with patch.object(deepseek_client.requests, "post", return_value=resp):
            with self.assertRaises(deepseek_client.DeepSeekAPIError):
                deepseek_client.chat("system", "user")


if __name__ == "__main__":
    unittest.main()
