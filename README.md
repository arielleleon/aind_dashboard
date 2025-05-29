# AIND Dashboard

## Features

### Conditional Row Highlighting

The DataTable now implements comprehensive conditional formatting to help users immediately identify subject performance based on alert status:

#### Color Schema
- **Orange Tones** (Performance Issues):
  - **SB (Severely Below)**: Dark orange highlighting with white text
  - **B (Below)**: Light orange highlighting with white text

- **Blue Tones** (Good Performance):
  - **G (Good)**: Light blue highlighting with white text  
  - **SG (Severely Good)**: Dark blue highlighting with white text

- **Brown Tone** (Threshold Alerts):
  - **T (Threshold Only)**: Brown highlighting for subjects with threshold alerts but no percentile category

- **Combined Alerts**: Enhanced styling with darker colors and additional borders for subjects with both percentile and threshold alerts (e.g., "SB, T")

#### Visual Elements
- **Severity Mapping**: Color brightness indicates severity (darker = more severe)
- **Border Indicators**: Left border in the alert color for visual consistency
- **Key Columns**: Alert-related columns (subject_id, combined_alert, percentile_category, overall_percentile) get stronger highlighting
- **Subtle Background**: Other columns receive very light background tinting
- **Hover Effects**: Smooth transitions and elevated appearance on row hover

#### Alert Categories
- **SB**: Severely Below (< 6.5 percentile)
- **B**: Below (6.5-28 percentile)  
- **N**: Normal (28-72 percentile) - no highlighting
- **G**: Good (72-93.5 percentile)
- **SG**: Severely Good (> 93.5 percentile)
- **NS**: Not Scored - no highlighting
- **T**: Threshold alerts for session counts or water intake

## Usage