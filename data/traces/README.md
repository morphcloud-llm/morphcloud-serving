# Spot market telemetry and interruption traces

The original empirical artifacts were not present in the supplied ZIP. Do not replace them with synthetic data under empirical filenames.

## Predictor training telemetry (Section 3.8)

The manuscript describes six months of telemetry (July–December 2024) from `us-east-1`, `us-west-2`, `us-central1`, and `europe-west1`: 6,220,800 samples, 3,847 distinct preemption events, and 11,541 positive 30-second-window samples.

For local/private training, place region CSVs under `data/traces/raw/` (Git-ignored) with columns:

```text
timestamp,spot_price,az_rejection_rate,n_active,uptime_s,preemption_event
```

## Evaluation traces (Section 4.1.2)

The manuscript describes a separate 90-day empirical interruption trace with 2,134 observed interruption signals used to construct the trace-driven injection experiment.

## Public release

Place only properly anonymized releasable event logs/traces under `data/traces/anonymized/`. The paper states that anonymized event logs and held-out prediction outputs will be released. Those files must come from the original experiment artifacts; this repository does not synthesize substitutes.
