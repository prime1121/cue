"""Offline checks for corruption, bounded downloads, and atomic preparation."""

import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_models as models


class Response(io.BytesIO):
    def __init__(self, body: bytes, headers=None, url="https://example.invalid/model.onnx"):
        super().__init__(body)
        self.headers = headers or {}
        self.url = url

    def geturl(self):
        return self.url


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.destination = Path(self.temporary.name)
        self.body = b"synthetic ONNX stand-in"
        self.model = models.Model("test.onnx", "test.onnx", hashlib.sha256(self.body).hexdigest())

    def leftovers(self):
        return list(self.destination.glob("*.partial"))

    def test_download_validates_and_reuses_without_network(self):
        with patch.object(models, "urlopen", return_value=Response(self.body)) as request:
            target = models.prepare(self.model, self.destination)
        self.assertEqual(target.read_bytes(), self.body)
        self.assertEqual(request.call_args.kwargs["timeout"], models.SOCKET_TIMEOUT_SECONDS)
        self.assertEqual(self.leftovers(), [])
        with patch.object(models, "urlopen", side_effect=AssertionError("unexpected network")):
            self.assertEqual(models.prepare(self.model, self.destination, check_only=True), target)

    def test_corrupt_existing_file_is_preserved(self):
        target = self.destination / self.model.name
        target.write_bytes(b"unexpected old bytes")
        with patch.object(models, "urlopen", side_effect=AssertionError("unexpected network")):
            with self.assertRaisesRegex(ValueError, "Existing checksum mismatch"):
                models.prepare(self.model, self.destination)
        self.assertEqual(target.read_bytes(), b"unexpected old bytes")

    def test_check_only_rejects_missing_without_network(self):
        with patch.object(models, "urlopen", side_effect=AssertionError("unexpected network")):
            with self.assertRaises(FileNotFoundError):
                models.prepare(self.model, self.destination, check_only=True)

    def test_bad_download_is_never_installed_and_partial_is_removed(self):
        with patch.object(models, "urlopen", return_value=Response(b"corrupt")):
            with self.assertRaisesRegex(ValueError, "Downloaded checksum mismatch"):
                models.prepare(self.model, self.destination)
        self.assertFalse((self.destination / self.model.name).exists())
        self.assertEqual(self.leftovers(), [])

    def test_oversized_header_is_rejected_before_download(self):
        response = Response(self.body, {"Content-Length": str(models.MAX_BYTES + 1)})
        with patch.object(models, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "Content-Length"):
                models.prepare(self.model, self.destination)
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_oversized_body_without_header_is_rejected(self):
        with patch.object(models, "MAX_BYTES", 4), patch.object(models, "urlopen", return_value=Response(self.body)):
            with self.assertRaisesRegex(ValueError, "download exceeds"):
                models.prepare(self.model, self.destination)
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_deadline_failure_discards_partial(self):
        with patch.object(models, "urlopen", return_value=Response(self.body)), patch.object(
            models.time, "monotonic", side_effect=[0.0, models.DOWNLOAD_TIMEOUT_SECONDS + 1]
        ):
            with self.assertRaises(TimeoutError):
                models.prepare(self.model, self.destination)
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_insecure_redirect_is_rejected(self):
        with patch.object(models, "urlopen", return_value=Response(self.body, url="http://example.invalid/model")):
            with self.assertRaisesRegex(ValueError, "insecure"):
                models.prepare(self.model, self.destination)
        self.assertEqual(list(self.destination.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
