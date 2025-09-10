from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator


router = APIRouter()


@router.post("/add")
def add(num1: int, num2: int):
    return {"sum": num1 + num2}


# Pydantic model for validating the input data
class Sample(BaseModel):
    operation: str = Field(..., pattern="^(sum|sub|mul|div)$")
    num1: int = Field(..., ge=0)
    num2: int = Field(..., ge=0)

    # validate that if operation is division, num2 is not 0
    @validator("num2")
    def num2_not_zero(cls, v: int, values: dict):
        if "operation" in values and values["operation"] == "div" and v == 0:
            raise ValueError("num2 cannot be 0 when operation is division")
        return v


@router.post("/operation")
def operate(sample: Sample):
    if sample.operation == "sum":
        return {"result": sample.num1 + sample.num2}
    elif sample.operation == "sub":
        return {"result": sample.num1 - sample.num2}
    elif sample.operation == "mul":
        return {"result": sample.num1 * sample.num2}
    elif sample.operation == "div":
        return {"result": sample.num1 / sample.num2}
    else:
        raise HTTPException(status_code=400, detail="Invalid operation")
