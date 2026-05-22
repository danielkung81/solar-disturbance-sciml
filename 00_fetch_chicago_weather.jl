"""
Fetch Chicago hourly weather from Open-Meteo's free historical-weather archive
and write a minimal CSV compatible with `load_real_weather` in shared_v3.jl.

Output columns: Timestamp, Building_ID, temp_out, shortwave_radiation, cloud_cover.

Usage:
    julia --project=. 00_fetch_chicago_weather.jl

The script is idempotent: if `data/chicago_weather.csv` already exists it does
nothing. Delete the file or pass `--force` to refetch.
"""

using HTTP, JSON3, CSV, DataFrames, Dates

const CHICAGO_LAT = 41.88
const CHICAGO_LON = -87.63
const START_DATE  = Date(2025, 1, 1)
const END_DATE    = Date(2025, 1, 14)
const OUT_PATH    = joinpath(@__DIR__, "data", "chicago_weather.csv")

function fetch_open_meteo(lat, lon, start_date, end_date)
    url = "https://archive-api.open-meteo.com/v1/archive"
    query = Dict(
        "latitude"    => string(lat),
        "longitude"   => string(lon),
        "start_date"  => string(start_date),
        "end_date"    => string(end_date),
        "hourly"      => "temperature_2m,shortwave_radiation,cloud_cover",
        "timezone"    => "America/Chicago",
    )
    response = HTTP.get(url; query = query)
    body = JSON3.read(String(response.body))
    return body.hourly
end

function main(; force::Bool = false)
    mkpath(dirname(OUT_PATH))
    if isfile(OUT_PATH) && !force
        println("$(OUT_PATH) already exists. Pass --force to refetch.")
        return
    end
    println("Fetching Open-Meteo weather: Chicago ($(CHICAGO_LAT), $(CHICAGO_LON)) " *
            "from $(START_DATE) to $(END_DATE) …")
    hourly = fetch_open_meteo(CHICAGO_LAT, CHICAGO_LON, START_DATE, END_DATE)

    n = length(hourly.time)
    df = DataFrame(
        Timestamp           = [Dates.format(DateTime(t), "yyyy-mm-dd HH:MM:SS")
                               for t in hourly.time],
        Building_ID         = fill("B001", n),
        temp_out            = Float64.(hourly.temperature_2m),
        shortwave_radiation = Float64.(hourly.shortwave_radiation),
        cloud_cover         = Float64.(hourly.cloud_cover),
    )
    CSV.write(OUT_PATH, df)
    println("Wrote $n hourly rows to $(OUT_PATH).")
end

force = "--force" in ARGS
main(; force = force)
