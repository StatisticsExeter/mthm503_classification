SELECT CASE 
         WHEN construction_age_estimated < 1930 THEN 'Pre-30s'
         WHEN construction_age_estimated >= 1930 THEN 'Post-30s'
         ELSE NULL
       END AS built_age,
       current_energy_efficiency, environment_impact_current, energy_consumption_current,
       co2_emissions_current, lighting_cost_current, heating_cost_current, hot_water_cost_current
FROM energy.energy_certificates 
WHERE local_authority = 'E08000033' AND tenure::text IN ('Owner occupied', 'Private Rented', 'Social Rented')  AND 
      property_type::text = 'House';
