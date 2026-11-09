import warnings
from io import BytesIO

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_TOTAL_BYTES = 40 * 1024 * 1024  # 40 MB
MAX_PIXELS = 25_000_000             # 25 megapixels

ALLOWED_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
}


async def validate_images(
    images: list[UploadFile],
) -> list[tuple[bytes, str]]:
    """Return verified image bytes and their detected MIME types."""
    if not 1 <= len(images) <= 5:
        raise HTTPException(
            status_code=422,
            detail="Upload between 1 and 5 photographs of the same tree.",
        )

    validated = []
    total_bytes = 0

    for number, upload in enumerate(images, start=1):
        # Read at most the limit plus one byte to detect oversized files.
        content = await upload.read(MAX_IMAGE_BYTES + 1)

        if len(content) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image {number} exceeds the 10 MB limit.",
            )

        if not content:
            raise HTTPException(
                status_code=422,
                detail=f"Image {number} is empty.",
            )

        total_bytes += len(content)
        if total_bytes > MAX_TOTAL_BYTES:
            raise HTTPException(
                status_code=413,
                detail="The combined image size exceeds 40 MB.",
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)

                with Image.open(BytesIO(content)) as image:
                    detected_mime = ALLOWED_FORMATS.get(image.format)
                    if detected_mime is None:
                        raise HTTPException(
                            status_code=415,
                            detail=f"Image {number} must be a JPEG or PNG.",
                        )

                    width, height = image.size
                    if width * height > MAX_PIXELS:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"Image {number} exceeds 25 megapixels. "
                                "Please reduce its dimensions."
                            ),
                        )

                    if getattr(image, "n_frames", 1) > 1:
                        raise HTTPException(
                            status_code=415,
                            detail=f"Image {number}: animated images are not supported.",
                        )

                    image.verify()

                # Reopen to check that the actual pixel data can be decoded.
                with Image.open(BytesIO(content)) as image:
                    image.load()

        except HTTPException:
            raise
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ):
            raise HTTPException(
                status_code=413,
                detail=f"Image {number} has excessive dimensions.",
            ) from None
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Image {number} cannot be read. "
                    "Please upload a valid JPEG or PNG."
                ),
            ) from None

        validated.append((content, detected_mime))

    return validated