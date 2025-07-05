# MNAR Missingness Embedding

Code for the paper:
Sun M, Engelhard MM, Bedoya AD, Goldstein BA. Incorporating informatively collected laboratory data from EHR in clinical prediction models. BMC Medical Informatics and Decision Making. 2024 Jul 24;24(1):206.

The code implement entity embedding for solving missingness in structured data and sequence data (i.e. EHR data)  
---

## Modules

### 1. `data_utils`:
In this module, we define classes that process stucture data from a csv file a `DataLoader` objects.

### 2. `nn_models`:
In this module, we define classes of MLP and LSTM models that are with/without the embedding layers.

### 3. `bootstrp`:
In this module, we define classes and features that calculate bootstrapped confidence interval using parallel computing


