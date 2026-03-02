def business_metrics(prediction_kwh_per_day, rate_per_kwh: float = 3.6, emission_kg_per_kwh: float = 0.82):
    daily_cost = prediction_kwh_per_day * rate_per_kwh
    monthly_cost = daily_cost * 30
    annual_cost = monthly_cost * 12
    annual_co2 = prediction_kwh_per_day * 365 * emission_kg_per_kwh
    if prediction_kwh_per_day <= 10:
        grade = "A"
        descriptor = "Low Consumption"
    elif prediction_kwh_per_day <= 15:
        grade = "B"
        descriptor = "Moderate Consumption"
    elif prediction_kwh_per_day <= 20:
        grade = "C"
        descriptor = "Above Average"
    else:
        grade = "D"
        descriptor = "High Consumption"
    return {
        "daily_cost": round(daily_cost, 2),
        "monthly_cost": round(monthly_cost, 2),
        "annual_cost": round(annual_cost, 2),
        "annual_co2_kg": round(annual_co2, 2),
        "efficiency_grade": grade,
        "grade_legend": "A ≤ 10, B ≤ 15, C ≤ 20, D > 20",
        "grade_descriptor": descriptor,
    }
