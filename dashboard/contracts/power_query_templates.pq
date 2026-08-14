// Create a TEXT parameter named pProjectRoot first.
// It must point to the root of the extracted project/handoff folder, never to a personal hard-coded path in query code.

// Query/function: fxProjectFile
(relativePath as text) as binary =>
let
    Normalized = Text.Replace(relativePath, "/", "\\"),
    FullPath = pProjectRoot & "\\" & Normalized,
    Source = File.Contents(FullPath)
in
    Source

// Query/function: fxGoldParquet
(relativePath as text) as table =>
let
    Source = Parquet.Document(fxProjectFile(relativePath))
in
    Source

// Example mart query:
// let Source = fxGoldParquet("data/gold/marts/mart_market_monthly.parquet") in Source

// Example dimension query:
// let Source = fxGoldParquet("data/gold/dimensions/dim_location.parquet") in Source

// dashboard_quality_status (CSV metadata used by P02):
// let
//     Source = Csv.Document(fxProjectFile("data/gold/metadata/dashboard_quality_status.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
//     Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true])
// in
//     Promoted
