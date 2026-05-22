"""
Track A — Synthetic ground-truth simulation.

Runs the four-model panel on N_SEEDS independent synthetic realisations and
reports temperature/solar/cloud RMSE with 95% CIs and paired DM tests.

Models:
  M1: Physics-only        (I_s = 0)
  M2: Geometric clear-sky (known I_clr from solar geometry, c = 0)
  M3: Learned RBF clear-sky (RBF fit by one-step prediction error, c = 0)
  M4: Proposed UKF with latent cloud state (RBF clear-sky + sigmoid-bounded c)
"""

include("shared_v3.jl")
using .SolarThermalV3
using CSV, DataFrames, Statistics, Random, Printf

ensure_output_dir()

const N_SEEDS    = 20
const N_DAYS     = 14
const DT_HOURS   = 0.25            # 15-min sampling
const TEST_DAYS  = 3               # last 3 days held out as test
const BUILDING   = DEFAULT_BUILDING

steps_per_day = round(Int, 24 / DT_HOURS)
n_total = N_DAYS * steps_per_day
test_start = n_total - TEST_DAYS * steps_per_day + 1
train_idx = 1:(test_start - 1)
test_idx  = test_start:n_total

println("Track A: $N_SEEDS seeds × $N_DAYS days @ $DT_HOURS h = $n_total steps each")
println("Train: 1..$(test_start-1)   Test: $(test_start)..$n_total")

per_seed = DataFrame()
err_T = Dict("M1" => Vector{Float64}(), "M2" => Vector{Float64}(),
             "M3" => Vector{Float64}(), "M4" => Vector{Float64}())
err_Is = Dict("M2" => Vector{Float64}(), "M3" => Vector{Float64}(),
              "M4" => Vector{Float64}())

example_traces = (df = nothing,)

for seed in 1:N_SEEDS
    df = simulate_track_a(; n_days = N_DAYS, dt_hours = DT_HOURS,
                            building = BUILDING, seed = seed)
    daylight = df.I_clr_true .> 30.0
    df_train = df[train_idx, :]

    # ----- M1: Physics-only -----
    T_M1 = rollout_physics_only(df, BUILDING, DT_HOURS)
    Is_M1 = zeros(nrow(df))

    # ----- M2: Geometric clear-sky (no cloud) -----
    I_clr_geo = df.I_clr_true                # in Track A we know the true geometric profile
    T_M2 = rollout_clearsky_only(df, I_clr_geo, BUILDING, DT_HOURS)
    Is_M2 = I_clr_geo                         # c assumed 0

    # ----- M3: Learned RBF clear-sky, no cloud at deployment -----
    # Offline ID uses training-time cloud cover (would come from weather service in practice).
    # Online deployment treats cloud as zero (no-cloud ablation of the proposed method).
    rbf_fit = fit_clear_sky_rbf_oneshot(df_train, BUILDING, DT_HOURS;
                                        c_train = df_train.c_true,
                                        n_basis = 8, ridge = 1e-3)
    phi_full, _ = rbf_matrix(df.hour_of_day; n_basis = 8)
    I_clr_rbf = phi_full * rbf_fit.weights
    T_M3 = rollout_clearsky_only(df, I_clr_rbf, BUILDING, DT_HOURS)
    Is_M3 = I_clr_rbf                         # c assumed 0

    # ----- M4: Proposed UKF with latent cloud -----
    ukf = run_ukf_latent_cloud(df, I_clr_rbf, BUILDING, DT_HOURS;
                                q_T = 0.05, q_z = 0.05, r_meas = 0.04,
                                lambda_revert = 0.02)
    T_M4  = ukf.temperature
    c_M4  = ukf.cloud
    Is_M4 = ukf.effective_solar

    # ----- Metrics (test set) -----
    function m(p, a)
        return (rmse = rmse(p, a), mae = mae(p, a))
    end
    T_true_test = df.T_true[test_idx]
    Is_true_test = df.I_s_true[test_idx]
    c_true_test  = df.c_true[test_idx]
    day_test = daylight[test_idx]

    rT_M1 = m(T_M1[test_idx], T_true_test)
    rT_M2 = m(T_M2[test_idx], T_true_test)
    rT_M3 = m(T_M3[test_idx], T_true_test)
    rT_M4 = m(T_M4[test_idx], T_true_test)

    rIs_M2 = m(Is_M2[test_idx][day_test], Is_true_test[day_test])
    rIs_M3 = m(Is_M3[test_idx][day_test], Is_true_test[day_test])
    rIs_M4 = m(Is_M4[test_idx][day_test], Is_true_test[day_test])

    rc_M4  = m(c_M4[test_idx][day_test],  c_true_test[day_test])

    push!(err_T["M1"], rT_M1.rmse); push!(err_T["M2"], rT_M2.rmse)
    push!(err_T["M3"], rT_M3.rmse); push!(err_T["M4"], rT_M4.rmse)
    push!(err_Is["M2"], rIs_M2.rmse); push!(err_Is["M3"], rIs_M3.rmse)
    push!(err_Is["M4"], rIs_M4.rmse)

    push!(per_seed, (
        seed = seed,
        T_M1_rmse = rT_M1.rmse, T_M2_rmse = rT_M2.rmse,
        T_M3_rmse = rT_M3.rmse, T_M4_rmse = rT_M4.rmse,
        T_M1_mae  = rT_M1.mae,  T_M2_mae  = rT_M2.mae,
        T_M3_mae  = rT_M3.mae,  T_M4_mae  = rT_M4.mae,
        Is_M2_rmse = rIs_M2.rmse, Is_M3_rmse = rIs_M3.rmse, Is_M4_rmse = rIs_M4.rmse,
        Is_M2_mae  = rIs_M2.mae,  Is_M3_mae  = rIs_M3.mae,  Is_M4_mae  = rIs_M4.mae,
        c_M4_rmse  = rc_M4.rmse,  c_M4_mae   = rc_M4.mae,
    ); promote = true)

    if seed == 1
        global example_traces = (
            df = df, T_M1 = T_M1, T_M2 = T_M2, T_M3 = T_M3, T_M4 = T_M4,
            c_M4 = c_M4, Is_M4 = Is_M4, I_clr_rbf = I_clr_rbf,
            test_idx = collect(test_idx)
        )
    end
end

# ----- Aggregate ----
function summarise(metric_col::Symbol)
    summary = DataFrame(model = String[], metric = String[], mean = Float64[],
                        ci_low = Float64[], ci_high = Float64[], std = Float64[], n = Int[])
    for col in [:T_M1_rmse, :T_M2_rmse, :T_M3_rmse, :T_M4_rmse,
                :T_M1_mae,  :T_M2_mae,  :T_M3_mae,  :T_M4_mae,
                :Is_M2_rmse, :Is_M3_rmse, :Is_M4_rmse,
                :Is_M2_mae,  :Is_M3_mae,  :Is_M4_mae,
                :c_M4_rmse,  :c_M4_mae]
        if col == metric_col
            s = mean_ci(per_seed[!, col])
            sname = String(col)
            push!(summary, (model = sname, metric = sname,
                            mean = s.mean, ci_low = s.ci_low, ci_high = s.ci_high,
                            std = s.std, n = s.n))
        end
    end
    return summary
end

agg_rows = NamedTuple[]
for col in [:T_M1_rmse, :T_M2_rmse, :T_M3_rmse, :T_M4_rmse,
            :T_M1_mae,  :T_M2_mae,  :T_M3_mae,  :T_M4_mae,
            :Is_M2_rmse, :Is_M3_rmse, :Is_M4_rmse,
            :Is_M2_mae,  :Is_M3_mae,  :Is_M4_mae,
            :c_M4_rmse,  :c_M4_mae]
    s = mean_ci(per_seed[!, col])
    push!(agg_rows, (metric = String(col),
                     mean = s.mean, ci_low = s.ci_low, ci_high = s.ci_high,
                     std = s.std, n = s.n))
end
agg = DataFrame(agg_rows)

# Paired DM tests: each baseline vs proposed (M4) on temperature RMSE per seed.
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

CSV.write(joinpath(OUTPUT_DIR, "track_a_per_seed.csv"), per_seed)
CSV.write(joinpath(OUTPUT_DIR, "track_a_aggregated.csv"), agg)
CSV.write(joinpath(OUTPUT_DIR, "track_a_dm_tests.csv"), dm_results)

# Save the first-seed traces for the figure script.
trace_df = DataFrame(
    step = 1:nrow(example_traces.df),
    hour = example_traces.df.hour,
    hour_of_day = example_traces.df.hour_of_day,
    T_true = example_traces.df.T_true,
    y_meas = example_traces.df.y_meas,
    T_M1 = example_traces.T_M1,
    T_M2 = example_traces.T_M2,
    T_M3 = example_traces.T_M3,
    T_M4 = example_traces.T_M4,
    c_true = example_traces.df.c_true,
    c_M4 = example_traces.c_M4,
    I_clr_true = example_traces.df.I_clr_true,
    I_clr_rbf  = example_traces.I_clr_rbf,
    I_s_true   = example_traces.df.I_s_true,
    Is_M4      = example_traces.Is_M4,
    in_test    = (1:nrow(example_traces.df)) .∈ Ref(test_idx),
)
CSV.write(joinpath(OUTPUT_DIR, "track_a_example_trace.csv"), trace_df)

println("\n--- Aggregated results (mean ± 95% CI, n=$N_SEEDS) ---")
for row in eachrow(agg)
    @printf "  %-14s = %8.4f  (95%% CI [%8.4f, %8.4f], std=%8.4f)\n" row.metric row.mean row.ci_low row.ci_high row.std
end

println("\n--- Paired DM tests vs proposed UKF (M4) ---")
for row in eachrow(dm_results)
    @printf "  %-30s mean_diff=%+8.3f  t=%+6.2f  p=%.4g\n" row.compare row.mean_diff row.t_stat row.p_value
end

println("\nTrack A complete. Outputs in $OUTPUT_DIR")
