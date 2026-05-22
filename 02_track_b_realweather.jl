"""
Track B — Real Chicago weather drives a simulated single-zone building.

Inputs:
  - Open-Meteo: real outdoor temperature, shortwave radiation, cloud cover
Outputs (simulated):
  - Indoor temperature trajectory from a known RC model
  - Ground-truth cloud transmittance derived from shortwave/I_clr_geometric

The Kaggle indoor temperature column is NOT used (it is random noise — see paper
revision notes). Track B validates that the proposed UKF can recover Open-Meteo's
real cloud cover from indoor temperature alone, without ever seeing the cloud_cover
column.
"""

include("shared_v3.jl")
using .SolarThermalV3
using CSV, DataFrames, Statistics, Random, Printf

ensure_output_dir()

const N_SEEDS    = 10                          # repeat with different measurement noise seeds
const N_DAYS     = 14
const DT_HOURS   = 1.0                         # weather data is hourly
const TEST_DAYS  = 3
const BUILDING   = DEFAULT_BUILDING
const WEATHER_FILE = joinpath(@__DIR__, "data", "chicago_weather.csv")

if !isfile(WEATHER_FILE)
    error("Weather CSV not found at $WEATHER_FILE.\n" *
          "Run `julia --project=. 00_fetch_chicago_weather.jl` first to download it from Open-Meteo.")
end

weather_df = SolarThermalV3.load_real_weather(WEATHER_FILE; building_id = "B001",
                                              n_days = N_DAYS)
n_total = nrow(weather_df)
steps_per_day = round(Int, 24 / DT_HOURS)
test_start = n_total - TEST_DAYS * steps_per_day + 1
train_idx = 1:(test_start - 1)
test_idx  = test_start:n_total

println("Track B: $N_SEEDS measurement-noise seeds × $n_total real weather hours")
println("Weather window: $(minimum(weather_df.ts))  →  $(maximum(weather_df.ts))")
println("Train: 1..$(test_start-1)   Test: $test_start..$n_total")

per_seed = DataFrame()
err_T = Dict("M1" => Vector{Float64}(), "M2" => Vector{Float64}(),
             "M3" => Vector{Float64}(), "M4" => Vector{Float64}())
err_Is = Dict("M2" => Vector{Float64}(), "M3" => Vector{Float64}(),
              "M4" => Vector{Float64}())

example_traces = (df = nothing,)

for seed in 1:N_SEEDS
    df = simulate_track_b(weather_df; building = BUILDING, dt_hours = DT_HOURS, seed = seed)

    daylight = df.I_clr_geo .> 30.0
    df_train = df[train_idx, :]

    # M1: Physics-only
    T_M1 = rollout_physics_only(df, BUILDING, DT_HOURS)

    # M2: Geometric clear-sky no cloud (uses solar geometry only)
    T_M2 = rollout_clearsky_only(df, df.I_clr_geo, BUILDING, DT_HOURS)
    Is_M2 = df.I_clr_geo

    # M3: RBF clear-sky fitted with known training-time cloud (from Open-Meteo)
    rbf_fit = fit_clear_sky_rbf_oneshot(df_train, BUILDING, DT_HOURS;
                                        c_train = df_train.c_true,
                                        n_basis = 8, ridge = 1e-3)
    phi_full, _ = rbf_matrix(df.hour_of_day; n_basis = 8)
    I_clr_rbf = phi_full * rbf_fit.weights
    T_M3 = rollout_clearsky_only(df, I_clr_rbf, BUILDING, DT_HOURS)
    Is_M3 = I_clr_rbf

    # M4: Proposed UKF with latent cloud — does NOT see the cloud column at test time
    ukf = run_ukf_latent_cloud(df, I_clr_rbf, BUILDING, DT_HOURS;
                                q_T = 0.05, q_z = 0.05, r_meas = 0.04,
                                lambda_revert = 0.02)
    T_M4 = ukf.temperature
    c_M4 = ukf.cloud
    Is_M4 = ukf.effective_solar

    # Metrics on test set
    T_true_test = df.T_true[test_idx]
    Is_true_test = df.I_s_true[test_idx]
    c_true_test  = df.c_true[test_idx]
    day_test = daylight[test_idx]

    rmse_T_M1 = rmse(T_M1[test_idx], T_true_test)
    rmse_T_M2 = rmse(T_M2[test_idx], T_true_test)
    rmse_T_M3 = rmse(T_M3[test_idx], T_true_test)
    rmse_T_M4 = rmse(T_M4[test_idx], T_true_test)

    rmse_Is_M2 = rmse(Is_M2[test_idx][day_test], Is_true_test[day_test])
    rmse_Is_M3 = rmse(Is_M3[test_idx][day_test], Is_true_test[day_test])
    rmse_Is_M4 = rmse(Is_M4[test_idx][day_test], Is_true_test[day_test])
    rmse_c_M4  = rmse(c_M4[test_idx][day_test],  c_true_test[day_test])

    push!(err_T["M1"], rmse_T_M1); push!(err_T["M2"], rmse_T_M2)
    push!(err_T["M3"], rmse_T_M3); push!(err_T["M4"], rmse_T_M4)
    push!(err_Is["M2"], rmse_Is_M2); push!(err_Is["M3"], rmse_Is_M3)
    push!(err_Is["M4"], rmse_Is_M4)

    push!(per_seed, (
        seed = seed,
        T_M1_rmse = rmse_T_M1, T_M2_rmse = rmse_T_M2,
        T_M3_rmse = rmse_T_M3, T_M4_rmse = rmse_T_M4,
        T_M1_mae  = mae(T_M1[test_idx], T_true_test),
        T_M2_mae  = mae(T_M2[test_idx], T_true_test),
        T_M3_mae  = mae(T_M3[test_idx], T_true_test),
        T_M4_mae  = mae(T_M4[test_idx], T_true_test),
        Is_M2_rmse = rmse_Is_M2, Is_M3_rmse = rmse_Is_M3, Is_M4_rmse = rmse_Is_M4,
        c_M4_rmse  = rmse_c_M4,
    ); promote = true)

    if seed == 1
        global example_traces = (
            df = df, T_M1 = T_M1, T_M2 = T_M2, T_M3 = T_M3, T_M4 = T_M4,
            c_M4 = c_M4, Is_M4 = Is_M4, I_clr_rbf = I_clr_rbf,
            test_idx = collect(test_idx)
        )
    end
end

agg_rows = NamedTuple[]
for col in [:T_M1_rmse, :T_M2_rmse, :T_M3_rmse, :T_M4_rmse,
            :T_M1_mae,  :T_M2_mae,  :T_M3_mae,  :T_M4_mae,
            :Is_M2_rmse, :Is_M3_rmse, :Is_M4_rmse,
            :c_M4_rmse]
    s = mean_ci(per_seed[!, col])
    push!(agg_rows, (metric = String(col),
                     mean = s.mean, ci_low = s.ci_low, ci_high = s.ci_high,
                     std = s.std, n = s.n))
end
agg = DataFrame(agg_rows)

dm_results = DataFrame()
for (name, vec) in err_T
    if name == "M4"; continue; end
    t = paired_dm_test(vec, err_T["M4"])
    push!(dm_results, (compare = "$(name) vs M4 on T_RMSE",
                       mean_diff = t.mean_diff, t_stat = t.t_stat, p_value = t.p_value,
                       n = t.n); promote = true)
end
for (name, vec) in err_Is
    if name == "M4"; continue; end
    t = paired_dm_test(vec, err_Is["M4"])
    push!(dm_results, (compare = "$(name) vs M4 on Is_RMSE",
                       mean_diff = t.mean_diff, t_stat = t.t_stat, p_value = t.p_value,
                       n = t.n); promote = true)
end

CSV.write(joinpath(OUTPUT_DIR, "track_b_per_seed.csv"), per_seed)
CSV.write(joinpath(OUTPUT_DIR, "track_b_aggregated.csv"), agg)
CSV.write(joinpath(OUTPUT_DIR, "track_b_dm_tests.csv"), dm_results)

trace_df = DataFrame(
    step = 1:nrow(example_traces.df),
    ts   = example_traces.df.ts,
    hour_of_day = example_traces.df.hour_of_day,
    T_true = example_traces.df.T_true,
    y_meas = example_traces.df.y_meas,
    T_M1 = example_traces.T_M1,
    T_M2 = example_traces.T_M2,
    T_M3 = example_traces.T_M3,
    T_M4 = example_traces.T_M4,
    c_true = example_traces.df.c_true,
    c_M4   = example_traces.c_M4,
    I_clr_geo = example_traces.df.I_clr_geo,
    I_clr_rbf = example_traces.I_clr_rbf,
    I_s_true  = example_traces.df.I_s_true,
    Is_M4     = example_traces.Is_M4,
    in_test   = (1:nrow(example_traces.df)) .∈ Ref(test_idx),
)
CSV.write(joinpath(OUTPUT_DIR, "track_b_example_trace.csv"), trace_df)

println("\n--- Aggregated results (mean ± 95% CI, n=$N_SEEDS) ---")
for row in eachrow(agg)
    @printf "  %-14s = %8.4f  (95%% CI [%8.4f, %8.4f], std=%8.4f)\n" row.metric row.mean row.ci_low row.ci_high row.std
end

println("\n--- Paired DM tests vs proposed UKF (M4) ---")
for row in eachrow(dm_results)
    @printf "  %-30s mean_diff=%+10.3f  t=%+6.2f  p=%.4g\n" row.compare row.mean_diff row.t_stat row.p_value
end

println("\nTrack B complete. Outputs in $OUTPUT_DIR")
