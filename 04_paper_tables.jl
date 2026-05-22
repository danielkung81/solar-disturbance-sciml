"""
Emit IEEE-conference-style LaTeX tables from the aggregated experiment CSVs.

Tables produced:
  table_2_track_a_temperature.tex     — Table II: temperature MAE/RMSE on Track A (synthetic)
  table_3_track_a_solar_cloud.tex     — Table III: solar and cloud RMSE on Track A
  table_4_track_b_summary.tex         — Table IV: Track B (real Chicago weather) summary
  table_5_dm_tests.tex                — Paired Diebold-Mariano test summary
  table_6_macros.tex                  — \\newcommand macros to drop into the paper preamble

Each table uses booktabs and IEEE column alignment so it can be \\input{} directly.
"""

include("shared_v3.jl")
using .SolarThermalV3
using CSV, DataFrames, Printf, Statistics

ensure_output_dir()

a_agg = CSV.read(joinpath(OUTPUT_DIR, "track_a_aggregated.csv"), DataFrame)
b_agg = CSV.read(joinpath(OUTPUT_DIR, "track_b_aggregated.csv"), DataFrame)
a_dm  = CSV.read(joinpath(OUTPUT_DIR, "track_a_dm_tests.csv"), DataFrame)
b_dm  = CSV.read(joinpath(OUTPUT_DIR, "track_b_dm_tests.csv"), DataFrame)

# Look up helper: row by :metric column
function pick(df::DataFrame, metric::AbstractString)
    row = filter(r -> r.metric == metric, df)
    @assert nrow(row) == 1 "metric $metric not found"
    return row[1, :]
end

fmt(x; digits=3) = @sprintf("%.*f", digits, x)
function ci_fmt(row; digits=3, tol=1e-9)
    half = (row.ci_high - row.ci_low) / 2
    if half < tol * max(abs(row.mean), 1.0)
        # deterministic row across seeds — no meaningful CI
        return fmt(row.mean; digits)
    else
        return @sprintf("%s [%s, %s]", fmt(row.mean; digits),
                        fmt(row.ci_low; digits), fmt(row.ci_high; digits))
    end
end

# Format p-value compactly
function pfmt(p::Real)
    if p < 1e-12
        return "\$<\\!10^{-12}\$"
    elseif p < 1e-4
        # use scientific
        s = @sprintf("%.1e", p)
        # turn 1.0e-08 into 1.0\times10^{-8}
        mant, expn = split(s, "e")
        return @sprintf("\$%s\\times10^{%d}\$", mant, parse(Int, expn))
    else
        return @sprintf("%.4g", p)
    end
end

# ============================================================
# Table II — Temperature MAE / RMSE on Track A
# ============================================================
let
    T1 = pick(a_agg, "T_M1_rmse"); T1m = pick(a_agg, "T_M1_mae")
    T2 = pick(a_agg, "T_M2_rmse"); T2m = pick(a_agg, "T_M2_mae")
    T3 = pick(a_agg, "T_M3_rmse"); T3m = pick(a_agg, "T_M3_mae")
    T4 = pick(a_agg, "T_M4_rmse"); T4m = pick(a_agg, "T_M4_mae")
    n = T1.n

    open(joinpath(OUTPUT_DIR, "table_2_track_a_temperature.tex"), "w") do io
        write(io, "% Track A: indoor-temperature estimation, n=$n seeds\n")
        write(io, "\\begin{table}[t]\n")
        write(io, "\\centering\n")
        write(io, "\\caption{Indoor-temperature estimation on Track A (synthetic ground truth, \$n=$n\$ seeds, 3-day test). Mean and 95\\% confidence interval across seeds.}\n")
        write(io, "\\label{tab:track_a_temperature}\n")
        write(io, "\\begin{tabular}{lcc}\n")
        write(io, "\\toprule\n")
        write(io, "Model & MAE (\$^\\circ\$C) & RMSE (\$^\\circ\$C) \\\\\n")
        write(io, "\\midrule\n")
        write(io, @sprintf("M1: Physics-only (\$I_s=0\$)        & %s & %s \\\\\n", ci_fmt(T1m; digits=2), ci_fmt(T1; digits=2)))
        write(io, @sprintf("M2: Geometric clear-sky, no cloud  & %s & %s \\\\\n", ci_fmt(T2m; digits=2), ci_fmt(T2; digits=2)))
        write(io, @sprintf("M3: Learned RBF clear-sky, no cloud& %s & %s \\\\\n", ci_fmt(T3m; digits=2), ci_fmt(T3; digits=2)))
        write(io, "\\midrule\n")
        write(io, @sprintf("\\textbf{M4: Proposed UKF (latent cloud)} & \\textbf{%s} & \\textbf{%s} \\\\\n", ci_fmt(T4m; digits=3), ci_fmt(T4; digits=3)))
        write(io, "\\bottomrule\n")
        write(io, "\\end{tabular}\n")
        write(io, "\\end{table}\n")
    end
end

# ============================================================
# Table III — Solar I_s and cloud RMSE on Track A
# ============================================================
let
    Is2 = pick(a_agg, "Is_M2_rmse")
    Is3 = pick(a_agg, "Is_M3_rmse")
    Is4 = pick(a_agg, "Is_M4_rmse")
    c4  = pick(a_agg, "c_M4_rmse")
    n = Is2.n

    open(joinpath(OUTPUT_DIR, "table_3_track_a_solar_cloud.tex"), "w") do io
        write(io, "% Track A: solar disturbance and cloud-cover estimation\n")
        write(io, "\\begin{table}[t]\n")
        write(io, "\\centering\n")
        write(io, "\\caption{Solar disturbance and latent cloud-cover estimation on Track A (daylight test samples, \$n=$n\$ seeds). M1 omits solar by construction, so \$I_s\$ is undefined.}\n")
        write(io, "\\label{tab:track_a_solar_cloud}\n")
        write(io, "\\begin{tabular}{lcc}\n")
        write(io, "\\toprule\n")
        write(io, "Model & \$I_s\$ RMSE (W/m\$^2\$) & Cloud RMSE \\\\\n")
        write(io, "\\midrule\n")
        write(io, @sprintf("M2: Geometric clear-sky, no cloud   & %s & --- \\\\\n", ci_fmt(Is2; digits=1)))
        write(io, @sprintf("M3: Learned RBF clear-sky, no cloud & %s & --- \\\\\n", ci_fmt(Is3; digits=1)))
        write(io, "\\midrule\n")
        write(io, @sprintf("\\textbf{M4: Proposed UKF (latent cloud)} & \\textbf{%s} & \\textbf{%s} \\\\\n",
                           ci_fmt(Is4; digits=1), ci_fmt(c4; digits=3)))
        write(io, "\\bottomrule\n")
        write(io, "\\end{tabular}\n")
        write(io, "\\end{table}\n")
    end
end

# ============================================================
# Table IV — Track B (real weather) summary
# ============================================================
let
    T1 = pick(b_agg, "T_M1_rmse"); T2 = pick(b_agg, "T_M2_rmse")
    T3 = pick(b_agg, "T_M3_rmse"); T4 = pick(b_agg, "T_M4_rmse")
    Is2 = pick(b_agg, "Is_M2_rmse"); Is3 = pick(b_agg, "Is_M3_rmse"); Is4 = pick(b_agg, "Is_M4_rmse")
    c4  = pick(b_agg, "c_M4_rmse")
    n = T1.n

    open(joinpath(OUTPUT_DIR, "table_4_track_b_summary.tex"), "w") do io
        write(io, "% Track B: real Chicago weather drives simulated building, n=$n noise seeds\n")
        write(io, "\\begin{table}[t]\n")
        write(io, "\\centering\n")
        write(io, "\\caption{Track B summary: real Chicago weather drives a simulated single-zone building. The UKF recovers Open-Meteo cloud cover from indoor temperature alone, without observing the cloud column. \$n=$n\$ measurement-noise seeds, 3-day test. Models M1 and M2 are deterministic given the weather and have no seed-to-seed variation.}\n")
        write(io, "\\label{tab:track_b_summary}\n")
        write(io, "\\begin{tabular}{lccc}\n")
        write(io, "\\toprule\n")
        write(io, "Model & T RMSE (\$^\\circ\$C) & \$I_s\$ RMSE (W/m\$^2\$) & Cloud RMSE \\\\\n")
        write(io, "\\midrule\n")
        write(io, @sprintf("M1: Physics-only                    & %s & ---   & ---   \\\\\n", ci_fmt(T1; digits=2)))
        write(io, @sprintf("M2: Geometric clear-sky, no cloud   & %s & %s   & ---   \\\\\n", ci_fmt(T2; digits=2), ci_fmt(Is2; digits=1)))
        write(io, @sprintf("M3: Learned RBF clear-sky, no cloud & %s & %s   & ---   \\\\\n", ci_fmt(T3; digits=2), ci_fmt(Is3; digits=1)))
        write(io, "\\midrule\n")
        write(io, @sprintf("\\textbf{M4: Proposed UKF} & \\textbf{%s} & \\textbf{%s} & \\textbf{%s} \\\\\n",
                           ci_fmt(T4; digits=3), ci_fmt(Is4; digits=1), ci_fmt(c4; digits=3)))
        write(io, "\\bottomrule\n")
        write(io, "\\end{tabular}\n")
        write(io, "\\end{table}\n")
    end
end

# ============================================================
# Table V — Paired DM-style significance tests
# ============================================================
let
    open(joinpath(OUTPUT_DIR, "table_5_dm_tests.tex"), "w") do io
        write(io, "% Paired Diebold-Mariano-style tests vs proposed UKF (M4)\n")
        write(io, "\\begin{table}[t]\n")
        write(io, "\\centering\n")
        write(io, "\\caption{Paired Diebold-Mariano-style significance tests between each baseline and the proposed UKF (M4). Squared-error differences are taken seed-by-seed; positive \$t\$ indicates that M4 has lower error.}\n")
        write(io, "\\label{tab:dm_tests}\n")
        write(io, "\\begin{tabular}{llcc}\n")
        write(io, "\\toprule\n")
        write(io, "Comparison & Metric & \$t\$-stat & \$p\$-value \\\\\n")
        write(io, "\\midrule\n")
        # Track A
        write(io, "\\multicolumn{4}{l}{\\emph{Track A (synthetic, \$n=20\$ seeds)}} \\\\\n")
        for row in eachrow(a_dm)
            tokens = split(row.compare, " on ")
            comparator = tokens[1]
            metric = replace(tokens[2], "T_RMSE" => "T", "Is_RMSE" => "\$I_s\$")
            write(io, @sprintf("%s & %s & %.2f & %s \\\\\n",
                               comparator, metric, row.t_stat, pfmt(row.p_value)))
        end
        write(io, "\\midrule\n")
        write(io, "\\multicolumn{4}{l}{\\emph{Track B (real weather, \$n=10\$ noise seeds)}} \\\\\n")
        for row in eachrow(b_dm)
            tokens = split(row.compare, " on ")
            comparator = tokens[1]
            metric = replace(tokens[2], "T_RMSE" => "T", "Is_RMSE" => "\$I_s\$")
            write(io, @sprintf("%s & %s & %.2f & %s \\\\\n",
                               comparator, metric, row.t_stat, pfmt(row.p_value)))
        end
        write(io, "\\bottomrule\n")
        write(io, "\\end{tabular}\n")
        write(io, "\\end{table}\n")
    end
end

# ============================================================
# Macros file for paper preamble
# ============================================================
let
    T4a = pick(a_agg, "T_M4_rmse"); T1a = pick(a_agg, "T_M1_rmse")
    T2a = pick(a_agg, "T_M2_rmse"); T3a = pick(a_agg, "T_M3_rmse")
    Is2a = pick(a_agg, "Is_M2_rmse"); Is3a = pick(a_agg, "Is_M3_rmse"); Is4a = pick(a_agg, "Is_M4_rmse")
    c4a  = pick(a_agg, "c_M4_rmse")

    T1b = pick(b_agg, "T_M1_rmse"); T2b = pick(b_agg, "T_M2_rmse")
    T3b = pick(b_agg, "T_M3_rmse"); T4b = pick(b_agg, "T_M4_rmse")
    Is2b = pick(b_agg, "Is_M2_rmse"); Is3b = pick(b_agg, "Is_M3_rmse"); Is4b = pick(b_agg, "Is_M4_rmse")
    c4b  = pick(b_agg, "c_M4_rmse")

    open(joinpath(OUTPUT_DIR, "table_6_macros.tex"), "w") do io
        write(io, "% Drop these \\newcommand definitions into the paper preamble so the abstract,\n")
        write(io, "% contributions list, and discussion can reference single sources of truth.\n")
        write(io, "% All numbers are mean across seeds (95% CIs are in the tables).\n\n")
        write(io, "% Track A (synthetic ground truth)\n")
        write(io, @sprintf("\\newcommand{\\TrackARmseTempPhysics}{%.2f}\n", T1a.mean))
        write(io, @sprintf("\\newcommand{\\TrackARmseTempGeoNoCloud}{%.2f}\n", T2a.mean))
        write(io, @sprintf("\\newcommand{\\TrackARmseTempRbfNoCloud}{%.2f}\n", T3a.mean))
        write(io, @sprintf("\\newcommand{\\TrackARmseTempProposed}{%.3f}\n", T4a.mean))
        write(io, @sprintf("\\newcommand{\\TrackARmseSolarGeoNoCloud}{%.1f}\n", Is2a.mean))
        write(io, @sprintf("\\newcommand{\\TrackARmseSolarRbfNoCloud}{%.1f}\n", Is3a.mean))
        write(io, @sprintf("\\newcommand{\\TrackARmseSolarProposed}{%.1f}\n", Is4a.mean))
        write(io, @sprintf("\\newcommand{\\TrackARmseCloudProposed}{%.3f}\n", c4a.mean))
        write(io, "\n% Track B (real Chicago weather drives simulated building)\n")
        write(io, @sprintf("\\newcommand{\\TrackBRmseTempPhysics}{%.2f}\n", T1b.mean))
        write(io, @sprintf("\\newcommand{\\TrackBRmseTempGeoNoCloud}{%.2f}\n", T2b.mean))
        write(io, @sprintf("\\newcommand{\\TrackBRmseTempRbfNoCloud}{%.2f}\n", T3b.mean))
        write(io, @sprintf("\\newcommand{\\TrackBRmseTempProposed}{%.3f}\n", T4b.mean))
        write(io, @sprintf("\\newcommand{\\TrackBRmseSolarGeoNoCloud}{%.1f}\n", Is2b.mean))
        write(io, @sprintf("\\newcommand{\\TrackBRmseSolarRbfNoCloud}{%.1f}\n", Is3b.mean))
        write(io, @sprintf("\\newcommand{\\TrackBRmseSolarProposed}{%.1f}\n", Is4b.mean))
        write(io, @sprintf("\\newcommand{\\TrackBRmseCloudProposed}{%.3f}\n", c4b.mean))
    end
end

println("LaTeX tables written:")
for fn in ["table_2_track_a_temperature.tex",
           "table_3_track_a_solar_cloud.tex",
           "table_4_track_b_summary.tex",
           "table_5_dm_tests.tex",
           "table_6_macros.tex"]
    println("  ", joinpath(OUTPUT_DIR, fn))
end
