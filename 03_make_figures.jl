"""
Generate publication-quality figures for the major-revision paper.

Reads the saved trace CSVs from Tracks A and B, produces:
  fig_temperature_estimation.png  — measured vs UKF estimate vs baselines
  fig_cloud_estimation.png        — true vs estimated cloud, with daylight shading
  fig_clear_sky_rbf.png           — true clear-sky vs learned RBF profile
  fig_solar_reconstruction.png    — true I_s vs UKF reconstruction
  fig_track_b_overview.png        — Track B (real weather) summary
"""

include("shared_v3.jl")
using .SolarThermalV3
using CSV, DataFrames, Plots, Statistics

ensure_output_dir()
default(fontfamily = "Helvetica", titlefontsize = 13, guidefontsize = 11,
        tickfontsize = 10, legendfontsize = 9, dpi = 200,
        framestyle = :box, gridalpha = 0.25)

track_a = CSV.read(joinpath(OUTPUT_DIR, "track_a_example_trace.csv"), DataFrame)
track_b = CSV.read(joinpath(OUTPUT_DIR, "track_b_example_trace.csv"), DataFrame)

# ---- Figure 1: Temperature estimation (Track A) ----
test_mask_a = track_a.in_test
hours_a = track_a.hour

p_temp = plot(hours_a, track_a.T_true, label = "True indoor T",
              lw = 2.0, color = :navy,
              xlabel = "Time (hours)", ylabel = "Indoor temperature (°C)",
              title = "Track A: Indoor temperature estimation (test set shaded)",
              size = (1200, 500), left_margin = 14Plots.mm, bottom_margin = 10Plots.mm)
plot!(p_temp, hours_a, track_a.y_meas, label = "Noisy measurement",
      color = :gray60, lw = 0.7, alpha = 0.55)
plot!(p_temp, hours_a, track_a.T_M1, label = "M1: Physics-only", lw = 1.5, color = :tomato, ls = :dot)
plot!(p_temp, hours_a, track_a.T_M3, label = "M3: RBF clear-sky (no cloud)", lw = 1.5, color = :orange, ls = :dash)
plot!(p_temp, hours_a, track_a.T_M4, label = "M4: Proposed UKF latent cloud",
      lw = 2.2, color = :forestgreen, ls = :solid)

# shade test region
test_start_h = hours_a[findfirst(test_mask_a)]
test_end_h   = hours_a[end]
vspan!(p_temp, [test_start_h, test_end_h], color = :lightgray, alpha = 0.18, label = "Test set")

savefig(p_temp, joinpath(OUTPUT_DIR, "fig_temperature_estimation.png"))
savefig(p_temp, joinpath(OUTPUT_DIR, "fig_temperature_estimation.pdf"))

# ---- Figure 2: Cloud cover estimation (Track A) ----
p_cloud = plot(hours_a, track_a.c_true, label = "True cloud cover",
               lw = 2.0, color = :navy,
               xlabel = "Time (hours)", ylabel = "Cloud cover c(t)",
               title = "Track A: Latent cloud cover estimation",
               ylims = (-0.02, 1.02), size = (1200, 450),
               left_margin = 14Plots.mm, bottom_margin = 10Plots.mm)
plot!(p_cloud, hours_a, track_a.c_M4, label = "UKF estimate ĉ(t)",
      lw = 1.8, color = :crimson, ls = :dash)
daylight_mask = track_a.I_clr_true .> 30.0
# Mark nighttime as light grey background ribbons (observability limitation)
night_idx = .!daylight_mask
function _night_bands(hours, mask)
    bands = Tuple{Float64,Float64}[]
    n = length(hours)
    i = 1
    while i <= n
        if mask[i]
            j = i
            while j <= n && mask[j]; j += 1; end
            push!(bands, (hours[i], hours[min(j, n)]))
            i = j
        else
            i += 1
        end
    end
    return bands
end

if any(night_idx)
    for (a, b) in _night_bands(hours_a, night_idx)
        vspan!(p_cloud, [a, b], color = :gray85, alpha = 0.45, label = "")
    end
end
vspan!(p_cloud, [test_start_h, test_end_h], color = :gold, alpha = 0.10, label = "Test set")

savefig(p_cloud, joinpath(OUTPUT_DIR, "fig_cloud_estimation.png"))
savefig(p_cloud, joinpath(OUTPUT_DIR, "fig_cloud_estimation.pdf"))

# ---- Figure 3: Clear-sky RBF profile (Track A) ----
# Aggregate by hour of day to show the canonical diurnal profile
df_byhour = combine(groupby(track_a, :hour_of_day),
                    :I_clr_true => mean => :I_clr_true_mean,
                    :I_clr_rbf  => mean => :I_clr_rbf_mean)
sort!(df_byhour, :hour_of_day)

p_rbf = plot(df_byhour.hour_of_day, df_byhour.I_clr_true_mean,
             label = "True clear-sky I_clr(t)", lw = 2.4, color = :navy,
             xlabel = "Hour of day", ylabel = "Clear-sky irradiance (W/m²)",
             title = "Track A: RBF approximation of clear-sky solar disturbance",
             size = (1000, 500), left_margin = 14Plots.mm, bottom_margin = 10Plots.mm,
             xticks = 0:3:24)
plot!(p_rbf, df_byhour.hour_of_day, df_byhour.I_clr_rbf_mean,
      label = "Learned RBF Î_clr(t; θ)", lw = 2.4, color = :crimson, ls = :dash)
savefig(p_rbf, joinpath(OUTPUT_DIR, "fig_clear_sky_rbf.png"))
savefig(p_rbf, joinpath(OUTPUT_DIR, "fig_clear_sky_rbf.pdf"))

# ---- Figure 4: Solar reconstruction (Track A) ----
p_sol = plot(hours_a, track_a.I_s_true, label = "True I_s(t) = (1-c)·I_clr",
             lw = 2.0, color = :navy,
             xlabel = "Time (hours)", ylabel = "Effective solar (W/m²)",
             title = "Track A: Effective solar disturbance reconstruction",
             size = (1200, 500), left_margin = 14Plots.mm, bottom_margin = 10Plots.mm)
plot!(p_sol, hours_a, track_a.Is_M4, label = "UKF reconstruction Î_s(t)",
      lw = 1.8, color = :crimson, ls = :dash)
plot!(p_sol, hours_a, track_a.I_clr_rbf, label = "Learned clear-sky (c=0)",
      lw = 1.3, color = :gray55, ls = :dot)
vspan!(p_sol, [test_start_h, test_end_h], color = :gold, alpha = 0.10, label = "Test set")
savefig(p_sol, joinpath(OUTPUT_DIR, "fig_solar_reconstruction.png"))
savefig(p_sol, joinpath(OUTPUT_DIR, "fig_solar_reconstruction.pdf"))

# ---- Figure 5: Track B overview (real Chicago weather) ----
n_b = nrow(track_b)
hours_b = collect(0:(n_b-1))
test_start_b = findfirst(track_b.in_test)
test_start_h_b = hours_b[test_start_b]
test_end_h_b   = hours_b[end]

p_b_T = plot(hours_b, track_b.T_true, label = "True T",
             lw = 2.0, color = :navy,
             xlabel = "Time (hours)", ylabel = "Indoor T (°C)",
             title = "Track B (real Chicago weather): indoor temperature",
             size = (1200, 350), left_margin = 14Plots.mm, bottom_margin = 8Plots.mm)
plot!(p_b_T, hours_b, track_b.T_M1, label = "M1 physics-only", color = :tomato, ls = :dot, lw = 1.4)
plot!(p_b_T, hours_b, track_b.T_M3, label = "M3 RBF no cloud", color = :orange, ls = :dash, lw = 1.4)
plot!(p_b_T, hours_b, track_b.T_M4, label = "M4 proposed UKF", color = :forestgreen, lw = 2.0)
vspan!(p_b_T, [test_start_h_b, test_end_h_b], color = :gold, alpha = 0.10, label = "Test set")

p_b_c = plot(hours_b, track_b.c_true, label = "True c (Open-Meteo)",
             lw = 2.0, color = :navy,
             xlabel = "Time (hours)", ylabel = "Cloud cover",
             title = "Track B: latent cloud-cover estimation",
             ylims = (-0.02, 1.02), size = (1200, 350),
             left_margin = 14Plots.mm, bottom_margin = 8Plots.mm)
plot!(p_b_c, hours_b, track_b.c_M4, label = "UKF estimate", color = :crimson, ls = :dash, lw = 1.6)
vspan!(p_b_c, [test_start_h_b, test_end_h_b], color = :gold, alpha = 0.10, label = "Test set")

p_b_Is = plot(hours_b, track_b.I_s_true, label = "True I_s",
              lw = 2.0, color = :navy,
              xlabel = "Time (hours)", ylabel = "Effective solar (W/m²)",
              title = "Track B: solar disturbance reconstruction",
              size = (1200, 350), left_margin = 14Plots.mm, bottom_margin = 8Plots.mm)
plot!(p_b_Is, hours_b, track_b.Is_M4, label = "UKF Î_s", color = :crimson, ls = :dash, lw = 1.6)
vspan!(p_b_Is, [test_start_h_b, test_end_h_b], color = :gold, alpha = 0.10, label = "Test set")

p_b = plot(p_b_T, p_b_c, p_b_Is, layout = (3, 1), size = (1200, 1000))
savefig(p_b, joinpath(OUTPUT_DIR, "fig_track_b_overview.png"))
savefig(p_b, joinpath(OUTPUT_DIR, "fig_track_b_overview.pdf"))

println("All figures written to $OUTPUT_DIR")
