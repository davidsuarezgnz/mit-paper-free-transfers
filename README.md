# MIT Sports Analytics Paper Competition

## Overview
This repository accompanies the abstract submission for the MIT Sports Analytics Paper Competition. As discussed with Kelly Condon (MIT responsible for the competition), due to the high volume of submissions, abstracts may be submitted with a placeholder GitHub link. The full implementation and code will be provided at the full paper deadline.

## Repository Structure (to be updated)
project/

├── data/ # Public datasets

├── notebooks/ # Reproducibility workflows

└── README.md # Project documentation

---

## Abstract
This paper formalizes an algorithmic framework for anticipating and ranking Bosman opportunities—players whose contracts expire within a configurable horizon—so clubs can negotiate 6–12 (up to 36) months ahead of the market. 

The system ingests contract end dates C_End, defines a detection window T_Window, and fuses (i) time-series market-value forecasts V_Future(t),  (ii) tactical fit scoring R_Match via multidimensional similarity d(x_i,x_ideal), and (iii) a risk index R_Risk (injury/instability) into a single Opportunity Index (OI) that drives ranked shortlists and alerts.
OIP=((V_Future(t)-V_Current)/V_Current)·R_Match·(1-d(x_i,x_ideal))·(1-(theta/T_Window))·1/(1+R_Risk)

At industry scale, the opportunity set is large and chronically under-exploited: >7,000 players finish contracts each year, with ≈38.7% of the market representing potential free-transfer cases; many clubs still track this manually. Our engine automates discovery and surfaces targets 12–36 months in advance, enabling pre-contracts and budget-efficient reinforcement. 

Empirical validation across European free-agent windows demonstrates the framework’s discriminatory power: top-decile OI candidates were associated with a 2.4× higher probability of positive net market revaluation within two seasons compared to the median, while minimizing expected wage inefficiencies by 18%. Sensitivity analysis shows that wage elasticity and market volatility dominate OI variance, underscoring the importance of accurate financial projections.

This research provides, to our knowledge, the first reproducible system for ranking free transfers through a unified optimization lens, bridging the gap between scouting heuristics and data-driven contract strategy. By embedding contract-window timing, legal eligibility, and financial risk directly into the model, the Opportunity Index transforms the traditionally opportunistic free-transfer market into a quantifiable decision domain. The resulting framework allows clubs not only to identify undervalued free agents, but also to strategically anticipate when Bosman opportunities align with long-term financial sustainability.
