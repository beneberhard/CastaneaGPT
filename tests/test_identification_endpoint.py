"""Multipart endpoint integration tests using real image validation and mocked services."""
import io
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.api.identification import router
from app.identification.schemas import IdentificationResult


class EndpointTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)
        image = io.BytesIO()
        Image.new('RGB', (8, 8), 'green').save(image, format='PNG')
        self.file = ('tree.png', image.getvalue(), 'image/png')

    @patch('app.api.identification.assess_identification')
    @patch('app.api.identification.identify_images')
    def test_valid_upload_with_optional_context(self, identify, assess):
        identify.return_value = IdentificationResult(status='no_result', message='No candidate')
        assess.return_value = {'summary': 'Unknown cause; please provide a close-up.', 'status': 'fallback'}
        response = self.client.post('/assess-images', files={'images': self.file})
        self.assertEqual(response.status_code, 200)
        self.assertIn('close-up', response.json()['assessment']['summary'])
        self.assertEqual(identify.call_args.kwargs['organ'], 'auto')
        self.assertEqual(assess.call_args.args[1].symptoms, '')
        self.assertEqual(assess.call_args.args[1].species, '')

    @patch('app.api.identification.identify_images')
    def test_invalid_image_rejected_before_provider_call(self, identify):
        response = self.client.post('/assess-images', files={'images': ('fake.jpg', b'hello', 'image/jpeg')})
        self.assertEqual(response.status_code, 422)
        identify.assert_not_called()

    def test_oversized_context_rejected(self):
        response = self.client.post('/assess-images', files={'images': self.file}, data={'location': 'x' * 301})
        self.assertEqual(response.status_code, 422)

    @patch('app.api.identification.assess_identification')
    @patch('app.api.identification.identify_images')
    def test_species_passes_to_assessment_and_response(self, identify, assess):
        identify.return_value = IdentificationResult(status='no_result', message='No candidate')
        assess.return_value = {'summary': 'Unknown cause', 'status': 'fallback'}
        response = self.client.post('/assess-images', files={'images': self.file},
                                    data={'species': 'Quercus robur'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['context']['species'], 'Quercus robur')
        self.assertEqual(assess.call_args.args[1].species, 'Quercus robur')

    def test_species_length_limit(self):
        response = self.client.post('/assess-images', files={'images': self.file},
                                    data={'species': 'x' * 201})
        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main()
