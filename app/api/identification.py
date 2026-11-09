from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.identification.validation import validate_images

from fastapi.concurrency import run_in_threadpool

from app.identification.plantnet import identify_images

from app.identification.assessment import(AssessmentContext, assess_identification)

router = APIRouter(tags=["Image assessment"])


@router.post("/assess-images")
async def assess_images(
    images: Annotated[
        list[UploadFile],
        File(
            json_schema_extra={
                "items": {"type": "string", "format": "binary"}
            }
        ),
    ],
    species: Annotated[str, Form(max_length=200)] = "",
    organ: Annotated[str, Form()] = "auto",
    location: Annotated[str, Form(max_length=300)] = "",
    observation_date: Annotated[str, Form(max_length=100)] = "",
    symptoms: Annotated[str, Form(max_length=3000)] = "",
):
    try:
        validated_images = await validate_images(images)

        allowed_organs = {
            "auto", "leaf", "bark", "branch", "fruit", "whole_tree"
        }
        if organ not in allowed_organs:
            raise HTTPException(
                status_code=422,
                detail="Unsupported plant organ.",
            )


        identification = await run_in_threadpool(
            identify_images,
            validated_images,
            organ=organ,
            language="en",
        )

        assessment_context = AssessmentContext(
            species=species,
            organ=organ,
            location=location,
            observation_date=observation_date,
            symptoms=symptoms,
            language="en",
        )

        assessment = await run_in_threadpool(
            assess_identification,
            identification,
            assessment_context,
        )



        return {
            "status": identification.status,
            "message": identification.message,
            "image_count": len(validated_images),
            "assessment": assessment,
            "identification": identification.model_dump(),
            "context": {
                "species": species,
                "organ": organ,
                "location": location,
                "observation_date": observation_date,
                "symptoms": symptoms,
            },
        }

        
    finally:
        for image in images:
            await image.close()