from pydantic import BaseModel, Field

class TransactionInput(BaseModel):
    amount: float = Field(..., gt=0)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    distance_from_home: float = Field(..., ge=0)
