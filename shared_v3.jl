module SolarThermalV3

using CSV
using DataFrames
using Dates
using LinearAlgebra
using Random
using Statistics
using Distributions

export
    OUTPUT_DIR,
    ensure_output_dir,
    BuildingParams,
    DEFAULT_BUILDING,
    HEAVY_MASS_BUILDING,
    sigmoid,
    logit,
    geometric_clearsky_profile,
    simulate_cloud_ou,
    simulate_track_a,
    simulate_track_b,
    rbf_matrix,
    fit_clear_sky_rbf_oneshot,
    fit_clear_sky_rbf_simulation,
    run_ukf_latent_cloud,
    simulate_2r2c,
    rmse, mae, mean_ci, paired_dm_test,
    rollout_physics_only,
    rollout_clearsky_only

const OUTPUT_DIR = joinpath(@__DIR__, "outputs")

ensure_output_dir() = (mkpath(OUTPUT_DIR); OUTPUT_DIR)

# ---- Building thermal parameters (1R1C single zone) ----
# Order-of-magnitude values chosen so that:
#   - tau = C_th / k_loss is on the order of a few hours (typical office room)
#   - A_w * I_clr_peak is roughly the same magnitude as k_loss * (T_set - T_ext)
#   - eta * P_h_peak is sufficient to maintain setpoint in winter
Base.@kwdef struct BuildingParams
    C_th::Float64       = 2.5e6          # J/K  thermal capacitance (~5h tau with k_loss=150)
    k_loss::Float64     = 150.0          # W/K  envelope conductance
    eta::Float64        = 0.95           # -    heater efficiency
    A_w::Float64        = 6.0            # m^2  effective window area
    T_set::Float64      = 21.0           # C    thermostat setpoint
    T_deadband::Float64 = 1.0            # C    on/off deadband
    P_h_max::Float64    = 3000.0         # W    heater capacity
end

const DEFAULT_BUILDING = BuildingParams()
const HEAVY_MASS_BUILDING = BuildingParams(C_th = 8.0e6)

# ---- Math helpers ----
sigmoid(x) = 1 / (1 + exp(-x))
logit(y)   = log(y / (1 - y))

rmse(p, a) = sqrt(mean((p .- a) .^ 2))
mae(p, a)  = mean(abs.(p .- a))

# Mean and 95% CI from a vector of per-seed metric values
function mean_ci(v::AbstractVector{<:Real}; alpha::Float64 = 0.05)
    n = length(v)
    m = mean(v)
    s = std(v)
    z = quantile(TDist(max(n - 1, 1)), 1 - alpha / 2)
    half = z * s / sqrt(n)
    return (mean = m, ci_low = m - half, ci_high = m + half, std = s, n = n)
end

# Diebold-Mariano-style paired test on squared-error sequences.
# Returns (mean_diff, t_stat, p_value) where positive mean_diff => model_a worse than model_b.
function paired_dm_test(err_a::AbstractVector{<:Real}, err_b::AbstractVector{<:Real})
    d = (err_a .^ 2) .- (err_b .^ 2)
    n = length(d)
    m = mean(d)
    s = std(d)
    t = m / (s / sqrt(n))
    # two-sided p-value from t distribution
    df = max(n - 1, 1)
    p = 2 * (1 - cdf(TDist(df), abs(t)))
    return (mean_diff = m, t_stat = t, p_value = p, n = n)
end

# ---- Geometric clear-sky model (simplified Bird/Hottel for a north-temperate site) ----
# I_clr(t) [W/m^2] as a function of time of day, day of year, latitude.
# Uses solar geometry + a static atmospheric transmittance.
function solar_declination(doy::Int)
    return deg2rad(23.45) * sin(2 * pi * (284 + doy) / 365)
end

function solar_zenith(t_hour::Float64, doy::Int, lat_deg::Float64)
    delta = solar_declination(doy)
    phi   = deg2rad(lat_deg)
    omega = deg2rad(15.0 * (t_hour - 12.0))
    cos_z = sin(phi) * sin(delta) + cos(phi) * cos(delta) * cos(omega)
    return max(cos_z, 0.0)
end

"""
    geometric_clearsky_profile(t_hour, doy, lat_deg; tau_atm=0.7, I_sc=1361.0)

Return clear-sky horizontal irradiance (W/m^2) using extraterrestrial * cos(zenith) * tau_atm.
Simple but physically grounded: serves as the geometric baseline R2 asked for.
"""
function geometric_clearsky_profile(t_hour::Float64, doy::Int, lat_deg::Float64;
                                    tau_atm::Float64 = 0.70, I_sc::Float64 = 1361.0)
    cz = solar_zenith(t_hour, doy, lat_deg)
    return I_sc * cz * tau_atm
end

# ---- Stochastic cloud cover: Ornstein-Uhlenbeck process on the logit ----
# Captures persistent weather state (sticky cloudy/clear periods) while staying in [0,1].
function simulate_cloud_ou(n::Int, dt_hours::Float64;
                           mean_cloud::Float64 = 0.45,
                           tau_hours::Float64 = 6.0,
                           sigma::Float64 = 0.6,
                           rng::AbstractRNG = Random.GLOBAL_RNG)
    z0 = logit(mean_cloud)
    z = zeros(n)
    z[1] = z0 + 0.5 * randn(rng)
    theta = 1.0 / tau_hours
    for k in 2:n
        z[k] = z[k-1] + theta * (z0 - z[k-1]) * dt_hours + sigma * sqrt(dt_hours) * randn(rng)
    end
    return sigmoid.(z)
end

# ---- Thermostatic heater (hysteretic on/off) ----
function thermostat_step(state::Bool, T::Float64, T_set::Float64, deadband::Float64)
    if state
        if T > T_set + 0.5 * deadband
            return false
        else
            return true
        end
    else
        if T < T_set - 0.5 * deadband
            return true
        else
            return false
        end
    end
end

# ---- Track A: pure synthetic simulation ----
# All states (T, c, I_s, I_clr) and parameters are known by construction.
function simulate_track_a(; n_days::Int = 14,
                            dt_hours::Float64 = 0.25,            # 15-min sampling
                            lat_deg::Float64 = 41.88,            # Chicago latitude
                            doy_start::Int = 15,                 # mid-January
                            building::BuildingParams = DEFAULT_BUILDING,
                            T_ext_amp::Float64 = 6.0,            # diurnal swing amplitude (C)
                            T_ext_mean::Float64 = -2.0,          # winter Chicago avg (C)
                            T_ext_noise::Float64 = 0.5,          # C
                            meas_noise_T::Float64 = 0.2,         # measurement noise std (C)
                            cloud_mean::Float64 = 0.45,
                            cloud_tau::Float64 = 6.0,
                            cloud_sigma::Float64 = 0.6,
                            seed::Int = 1)
    rng = MersenneTwister(seed)
    steps_per_day = round(Int, 24 / dt_hours)
    n = n_days * steps_per_day

    t_hours = collect(0:(n-1)) .* dt_hours
    t_of_day = mod.(t_hours, 24.0)
    doy = doy_start .+ (t_hours .÷ 24)

    # Outdoor temperature: diurnal cosine + low-frequency drift + AR(1) noise
    Text = T_ext_mean .+ T_ext_amp .* cos.(2 * pi .* (t_of_day .- 15) ./ 24)
    drift = cumsum(randn(rng, n)) .* 0.05
    drift .-= mean(drift)
    Text .+= drift
    eps_text = zeros(n); for k in 2:n; eps_text[k] = 0.6 * eps_text[k-1] + T_ext_noise * randn(rng); end
    Text .+= eps_text

    # Cloud cover OU and clear-sky profile
    c_true = simulate_cloud_ou(n, dt_hours;
                               mean_cloud = cloud_mean,
                               tau_hours = cloud_tau,
                               sigma = cloud_sigma,
                               rng = rng)
    I_clr_true = [geometric_clearsky_profile(t_of_day[k], Int(doy[k]), lat_deg) for k in 1:n]
    I_s_true   = (1 .- c_true) .* I_clr_true

    # Forward-integrate the 1R1C model with thermostatic heater
    T  = zeros(n); T[1] = building.T_set
    Ph = zeros(n)
    heater_on = false
    dt_sec = dt_hours * 3600.0
    for k in 1:(n-1)
        heater_on = thermostat_step(heater_on, T[k], building.T_set, building.T_deadband)
        Ph[k] = heater_on ? building.P_h_max : 0.0
        dT = (1 / building.C_th) * (
            -building.k_loss * (T[k] - Text[k]) +
            building.eta * Ph[k] +
            building.A_w * I_s_true[k]
        )
        T[k+1] = T[k] + dT * dt_sec
    end
    Ph[n] = heater_on ? building.P_h_max : 0.0

    y_meas = T .+ meas_noise_T .* randn(rng, n)

    return DataFrame(
        step       = 1:n,
        hour       = t_hours,
        hour_of_day= t_of_day,
        doy        = Int.(doy),
        T_ext      = Text,
        P_h        = Ph,
        c_true     = c_true,
        I_clr_true = I_clr_true,
        I_s_true   = I_s_true,
        T_true     = T,
        y_meas     = y_meas,
    )
end

# ---- Track B: real Chicago weather drives a simulated single-zone building ----
# We import real Open-Meteo weather (T_ext, shortwave, cloud_cover), define c_true from
# shortwave / I_clr_geometric, and simulate the building forward. The Kaggle indoor T is unused.
function load_real_weather(path::String; building_id::String = "B001", n_days::Int = 14)
    df = CSV.read(path, DataFrame)
    df.ts = DateTime.(df.Timestamp, dateformat"yyyy-mm-dd HH:MM:SS")
    df = subset(df, :Building_ID => ByRow(x -> string(x) == building_id))
    sort!(df, :ts)
    t0 = minimum(df.ts)
    df = subset(df, :ts => ByRow(t -> t < t0 + Day(n_days)))
    return df
end

function simulate_track_b(weather_df::DataFrame;
                          lat_deg::Float64 = 41.88,
                          building::BuildingParams = DEFAULT_BUILDING,
                          meas_noise_T::Float64 = 0.2,
                          dt_hours::Float64 = 1.0,
                          seed::Int = 1)
    rng = MersenneTwister(seed)
    n = nrow(weather_df)
    Text = Float64.(weather_df.temp_out)
    sw   = Float64.(weather_df.shortwave_radiation)
    # Day of year and time of day
    doy   = [dayofyear(t) for t in weather_df.ts]
    tod   = [hour(t) + minute(t)/60 for t in weather_df.ts]
    # Geometric clear-sky baseline
    I_clr_geo = [geometric_clearsky_profile(tod[k], doy[k], lat_deg) for k in 1:n]
    # Reverse-engineer the "true" cloud transmittance from real shortwave
    daylight  = I_clr_geo .> 30.0
    c_true    = ones(n) .* 0.5
    @inbounds for k in 1:n
        if daylight[k]
            c_true[k] = clamp(1.0 - sw[k] / max(I_clr_geo[k], 1e-6), 0.0, 1.0)
        else
            c_true[k] = NaN   # cloud is not identifiable at night
        end
    end
    # Smooth-fill nighttime with the nearest daytime value so the simulator has a value to use
    last_valid = 0.5
    c_filled = similar(c_true)
    @inbounds for k in 1:n
        if !isnan(c_true[k])
            last_valid = c_true[k]
            c_filled[k] = c_true[k]
        else
            c_filled[k] = last_valid
        end
    end

    I_s_true = (1 .- c_filled) .* I_clr_geo

    # Forward-integrate building
    T  = zeros(n); T[1] = building.T_set
    Ph = zeros(n)
    heater_on = false
    dt_sec = dt_hours * 3600.0
    for k in 1:(n-1)
        heater_on = thermostat_step(heater_on, T[k], building.T_set, building.T_deadband)
        Ph[k] = heater_on ? building.P_h_max : 0.0
        dT = (1 / building.C_th) * (
            -building.k_loss * (T[k] - Text[k]) +
            building.eta * Ph[k] +
            building.A_w * I_s_true[k]
        )
        T[k+1] = T[k] + dT * dt_sec
    end
    Ph[n] = heater_on ? building.P_h_max : 0.0

    y_meas = T .+ meas_noise_T .* randn(rng, n)

    return DataFrame(
        step       = 1:n,
        ts         = weather_df.ts,
        hour_of_day= tod,
        doy        = doy,
        T_ext      = Text,
        P_h        = Ph,
        c_true     = c_filled,
        c_obs_valid = .!isnan.(c_true),
        I_clr_geo  = I_clr_geo,
        I_s_true   = I_s_true,
        T_true     = T,
        y_meas     = y_meas,
    )
end

# ---- RBF basis over hour of day ----
function rbf_matrix(hours::AbstractVector{<:Real};
                    n_basis::Int = 8,
                    start_hour::Float64 = 5.0,
                    end_hour::Float64   = 19.0,
                    width::Float64      = 1.5)
    centers = collect(LinRange(start_hour, end_hour, n_basis))
    phi = zeros(length(hours), n_basis)
    @inbounds for j in 1:n_basis
        c = centers[j]
        @views phi[:, j] .= exp.(-((hours .- c) ./ width) .^ 2)
    end
    return phi, centers
end

"""
    fit_clear_sky_rbf_simulation(hour_of_day, I_clr_observed, daylight_mask; ridge)

Used when we have a direct (possibly cloud-corrected) clear-sky target.
Fits theta ≥ 0 RBF weights by weighted least squares on daylight samples.
"""
function fit_clear_sky_rbf_simulation(hour_of_day::AbstractVector{<:Real},
                                      target::AbstractVector{<:Real},
                                      daylight::AbstractVector{Bool};
                                      n_basis::Int = 8,
                                      ridge::Float64 = 1e-2)
    phi, centers = rbf_matrix(hour_of_day; n_basis = n_basis)
    w = Float64.(daylight)
    W = Diagonal(w)
    theta = (phi' * W * phi + ridge * I) \ (phi' * W * target)
    theta = max.(theta, 0.0)
    return (weights = theta, centers = centers, phi = phi, profile = phi * theta)
end

"""
    fit_clear_sky_rbf_oneshot(df, building, dt_hours; c_train, n_basis, ridge)

Prediction-error fit for the RBF clear-sky profile.

Offline parameter identification uses training-time cloud cover `c_train` (typically
obtained from a weather service such as Open-Meteo) as a known input, so the
identification equation is

    T_{k+1} - T_k - (dt/C) * [-k_loss * (T_ext - T) + eta * P_h]
        = (dt/C) * A_w * (1 - c_train_k) * phi_k * theta

The fitted RBF profile I_clr(t; theta) = phi(t) * theta represents the cloud-free
solar heat gain. At deployment time the UKF treats cloud as a latent state — this
is the realistic separation between offline ID and online estimation.

Pass `c_train = zeros(n)` to recover the older "no-cloud" prediction-error fit
used as an ablation baseline.
"""
function fit_clear_sky_rbf_oneshot(df::DataFrame, building::BuildingParams, dt_hours::Float64;
                                   c_train::Union{Nothing,AbstractVector{<:Real}} = nothing,
                                   n_basis::Int = 8, ridge::Float64 = 1e-3)
    phi, centers = rbf_matrix(df.hour_of_day; n_basis = n_basis)
    n = nrow(df)
    dt_sec = dt_hours * 3600.0
    T = df.y_meas
    rhs = T[2:end] .- T[1:end-1] .- (dt_sec / building.C_th) .* (
        -building.k_loss .* (T[1:end-1] .- df.T_ext[1:end-1]) .+
        building.eta .* df.P_h[1:end-1]
    )
    coef = (dt_sec / building.C_th) * building.A_w
    # Solve in scaled variables `eta_scaled = coef * theta` so the ridge term is
    # dimensionally comparable to phi'phi instead of being dominated by `coef^2`.
    cloud_input = c_train === nothing ? ones(n - 1) : (1 .- c_train[1:end-1])
    if c_train === nothing
        # Cloud-free ablation: assume c=0 during training (will recover an
        # attenuated effective clear-sky profile).
        cloud_input .= 1.0
    end
    A_scaled = phi[1:end-1, :] .* cloud_input
    # Scale-invariant ridge proportional to mean diagonal of A'A
    diagAA = [sum(A_scaled[:, j] .^ 2) for j in 1:n_basis]
    lam = ridge * (mean(diagAA) + 1e-12)
    eta_scaled = (A_scaled' * A_scaled + lam * I) \ (A_scaled' * rhs)
    theta = max.(eta_scaled ./ coef, 0.0)
    return (weights = theta, centers = centers, phi = phi, profile = phi * theta,
            scaled_ridge = lam, coef = coef)
end

# ---- Baseline 1: physics-only rollout (I_s = 0) ----
function rollout_physics_only(df::DataFrame, building::BuildingParams, dt_hours::Float64; T0::Float64 = NaN)
    n = nrow(df)
    T = zeros(n)
    T[1] = isnan(T0) ? df.y_meas[1] : T0
    dt_sec = dt_hours * 3600.0
    for k in 1:(n-1)
        dT = (1 / building.C_th) * (
            -building.k_loss * (T[k] - df.T_ext[k]) +
            building.eta * df.P_h[k]
        )
        T[k+1] = T[k] + dT * dt_sec
    end
    return T
end

# ---- Baseline: rollout given any clear-sky profile, no cloud (c=0) ----
function rollout_clearsky_only(df::DataFrame, I_clr::AbstractVector{<:Real},
                                building::BuildingParams, dt_hours::Float64; T0::Float64 = NaN)
    n = nrow(df)
    T = zeros(n)
    T[1] = isnan(T0) ? df.y_meas[1] : T0
    dt_sec = dt_hours * 3600.0
    for k in 1:(n-1)
        dT = (1 / building.C_th) * (
            -building.k_loss * (T[k] - df.T_ext[k]) +
            building.eta * df.P_h[k] +
            building.A_w * I_clr[k]
        )
        T[k+1] = T[k] + dT * dt_sec
    end
    return T
end

# ---- Proposed: UKF with augmented state [T, z = logit c] ----
function ukf_sigma_points(x::Vector{Float64}, P::Matrix{Float64}, lambda::Float64)
    n = length(x)
    M = Symmetric((n + lambda) .* P + 1e-9 * I(n))
    L = cholesky(M).L
    sigma = Matrix{Float64}(undef, n, 2n + 1)
    sigma[:, 1] = x
    @inbounds for j in 1:n
        sigma[:, j+1]     = x + L[:, j]
        sigma[:, n+j+1]   = x - L[:, j]
    end
    return sigma
end

"""
    run_ukf_latent_cloud(df, I_clr, building, dt_hours; ...)

UKF that estimates (T, z = logit(c)) from noisy temperature measurements alone.
df must have :T_ext, :P_h, :y_meas. I_clr is the learned clear-sky profile.
"""
function run_ukf_latent_cloud(df::DataFrame, I_clr::AbstractVector{<:Real},
                              building::BuildingParams, dt_hours::Float64;
                              q_T::Float64 = 0.05,
                              q_z::Float64 = 0.02,
                              r_meas::Float64 = 0.04,
                              z_mean::Float64 = 0.0,
                              lambda_revert::Float64 = 0.02,
                              alpha::Float64 = 0.5,
                              beta::Float64  = 2.0,
                              kappa::Float64 = 0.0,
                              T0::Float64 = NaN,
                              c0::Float64 = 0.5)
    n = nrow(df)
    state_dim = 2
    dt_sec = dt_hours * 3600.0
    lam = alpha^2 * (state_dim + kappa) - state_dim
    wm = fill(1 / (2 * (state_dim + lam)), 2state_dim + 1)
    wc = copy(wm)
    wm[1] = lam / (state_dim + lam)
    wc[1] = wm[1] + (1 - alpha^2 + beta)

    T_est  = zeros(n)
    c_est  = zeros(n)
    Is_est = zeros(n)

    x = [isnan(T0) ? df.y_meas[1] : T0, logit(clamp(c0, 1e-3, 1-1e-3))]
    P = Matrix{Float64}(I, 2, 2) .* 1.0
    Q = Diagonal([q_T, q_z])

    T_est[1]  = x[1]
    c_est[1]  = sigmoid(x[2])
    Is_est[1] = (1 - c_est[1]) * I_clr[1]

    for k in 1:(n-1)
        sigma = ukf_sigma_points(x, P, lam)
        sigma_pred = similar(sigma)
        @inbounds for j in 1:size(sigma, 2)
            Tk = sigma[1, j]
            zk = sigma[2, j]
            ck = sigmoid(zk)
            sigma_pred[1, j] = Tk + (dt_sec / building.C_th) * (
                -building.k_loss * (Tk - df.T_ext[k]) +
                building.eta * df.P_h[k] +
                building.A_w * (1 - ck) * I_clr[k]
            )
            sigma_pred[2, j] = zk + lambda_revert * (z_mean - zk) * dt_hours
        end

        x_pred = sigma_pred * wm
        P_pred = Matrix(Q)
        @inbounds for j in 1:size(sigma_pred, 2)
            d = sigma_pred[:, j] - x_pred
            P_pred += wc[j] .* (d * d')
        end

        z_sigma = sigma_pred[1, :]
        y_pred  = dot(wm, z_sigma)
        S = r_meas
        Pxz = zeros(state_dim)
        @inbounds for j in 1:length(z_sigma)
            xd = sigma_pred[:, j] - x_pred
            zd = z_sigma[j] - y_pred
            S  += wc[j] * zd * zd
            Pxz += wc[j] .* xd .* zd
        end

        Kgain = Pxz ./ S
        innov = df.y_meas[k+1] - y_pred
        x = x_pred + Kgain .* innov
        P = P_pred - S .* (Kgain * Kgain')
        P = Matrix(Symmetric(P)) + 1e-9 * I

        T_est[k+1]  = x[1]
        c_est[k+1]  = sigmoid(x[2])
        Is_est[k+1] = (1 - c_est[k+1]) * I_clr[k+1]
    end

    return (temperature = T_est, cloud = c_est, effective_solar = Is_est)
end

# ---- 2R2C robustness model (envelope mass + air node) ----
function simulate_2r2c(; n_days::Int = 14,
                         dt_hours::Float64 = 0.25,
                         lat_deg::Float64 = 41.88,
                         doy_start::Int = 15,
                         C_air::Float64 = 5.0e5,
                         C_env::Float64 = 4.0e6,
                         k_env::Float64 = 80.0,    # air <-> envelope
                         k_ext::Float64 = 120.0,   # envelope <-> outside
                         eta::Float64 = 0.95,
                         A_w::Float64 = 6.0,
                         T_set::Float64 = 21.0,
                         T_deadband::Float64 = 1.0,
                         P_h_max::Float64 = 3000.0,
                         seed::Int = 1,
                         cloud_mean::Float64 = 0.45,
                         cloud_tau::Float64 = 6.0,
                         cloud_sigma::Float64 = 0.6,
                         meas_noise_T::Float64 = 0.2)
    rng = MersenneTwister(seed)
    steps_per_day = round(Int, 24 / dt_hours)
    n = n_days * steps_per_day
    t_hours = collect(0:(n-1)) .* dt_hours
    t_of_day = mod.(t_hours, 24.0)
    doy = doy_start .+ (t_hours .÷ 24)

    Text = -2.0 .+ 6.0 .* cos.(2 * pi .* (t_of_day .- 15) ./ 24) .+ 0.5 .* randn(rng, n)

    c_true = simulate_cloud_ou(n, dt_hours;
                               mean_cloud = cloud_mean,
                               tau_hours = cloud_tau,
                               sigma = cloud_sigma,
                               rng = rng)
    I_clr_true = [geometric_clearsky_profile(t_of_day[k], Int(doy[k]), lat_deg) for k in 1:n]
    I_s_true   = (1 .- c_true) .* I_clr_true

    T_air = zeros(n); T_air[1] = T_set
    T_env = zeros(n); T_env[1] = (T_set + Text[1]) / 2
    Ph    = zeros(n)
    heater_on = false
    dt_sec = dt_hours * 3600.0
    for k in 1:(n-1)
        heater_on = thermostat_step(heater_on, T_air[k], T_set, T_deadband)
        Ph[k] = heater_on ? P_h_max : 0.0
        # air node
        dT_air = (1 / C_air) * (
            -k_env * (T_air[k] - T_env[k]) +
            eta * Ph[k] +
            A_w * I_s_true[k]
        )
        # envelope node
        dT_env = (1 / C_env) * (
            -k_ext * (T_env[k] - Text[k]) +
            k_env * (T_air[k] - T_env[k])
        )
        T_air[k+1] = T_air[k] + dT_air * dt_sec
        T_env[k+1] = T_env[k] + dT_env * dt_sec
    end
    Ph[n] = heater_on ? P_h_max : 0.0
    y_meas = T_air .+ meas_noise_T .* randn(rng, n)

    return DataFrame(
        step       = 1:n,
        hour       = t_hours,
        hour_of_day= t_of_day,
        doy        = Int.(doy),
        T_ext      = Text,
        P_h        = Ph,
        c_true     = c_true,
        I_clr_true = I_clr_true,
        I_s_true   = I_s_true,
        T_air_true = T_air,
        T_env_true = T_env,
        y_meas     = y_meas,
    )
end

end # module
