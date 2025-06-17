# Import shared utilities (replaces multiple individual imports)
from callbacks.shared_callback_utils import (
    Input,
    Output,
    State,
    app_tooltip,
    app_utils,
    callback,
    clientside_callback,
)

# Component instance is now shared from utilities - no need to re-initialize
# app_tooltip = AppHoverTooltip(app_utils=app_utils)  # Removed - using shared instance

# Tooltip Callbacks


# Server callback to update tooltip content
@callback(
    Output("subject-hover-tooltip", "children"),
    [Input("session-table", "active_cell")],
    [State("session-table", "data")],
)
def update_tooltip_content(active_cell, table_data):
    """
    Update tooltip content when hovering over subject ID cells
    We'll use the active_cell as a proxy since it's triggered by cell interactions

    Returns empty tooltip since we'll handle the actual tooltip display via clientside
    """
    return app_tooltip.get_empty_tooltip()


# Clientside callback for complete tooltip functionality
clientside_callback(
    """
    function(table_data, app_utils_placeholder) {
        // Initialize tooltip functionality
        console.log('Setting up tooltip functionality...');

        const table = document.getElementById('session-table');
        const tooltip = document.getElementById('subject-hover-tooltip');

        if (!table || !tooltip) {
            console.log('Table or tooltip not found');
            return window.dash_clientside.no_update;
        }

        let currentHoveredSubject = null;
        let hideTimeout = null;

        // Cache for tooltip data to avoid repeated server calls
        const tooltipDataCache = {};

        // Function to get alert color based on category
        function getAlertColor(category) {
            const colors = {
                'SB': '#FF6B35',  // Dark orange (Severely Below)
                'B': '#FFB366',   // Light orange (Below)
                'G': '#4A90E2',   // Light blue (Good)
                'SG': '#2E5A87',  // Dark blue (Severely Good)
                'T': '#795548'    // Brown (Threshold alerts)
            };
            return colors[category] || null;
        }

        // Function to create tooltip HTML content
        function createTooltipContent(subjectData) {
            if (!subjectData) return '';

            let html = '<div class="tooltip-content">';

            // Header
            html += '<div class="tooltip-header">';
            html += `<div class="tooltip-subject-id">${subjectData.subject_id}</div>`;
            html += `<div class="tooltip-strata">Strata: ${subjectData.strata_abbr || 'N/A'}</div>`;
            html += '</div>';

            // Overall percentile
            const overallPercentile = subjectData.session_overall_percentile || subjectData.overall_percentile;
            const overallCategory = subjectData.overall_percentile_category || 'NS';

            html += '<div class="tooltip-overall">';
            html += '<span class="tooltip-label">Overall: </span>';

            if (overallPercentile != null && !isNaN(overallPercentile)) {
                const color = getAlertColor(overallCategory);
                const colorStyle = color ? `color: ${color}; font-weight: 600;` : 'font-weight: 600;';
                html += `<span class="tooltip-value" style="${colorStyle}">${overallPercentile.toFixed(1)}%</span>`;
            } else {
                html += '<span class="tooltip-value tooltip-ns">Not Scored</span>';
            }
            html += '</div>';

            // Active feature alerts
            const features = ['finished_trials', 'ignore_rate', 'total_trials', 'foraging_performance', 'abs(bias_naive)'];
            const featureNames = {
                'finished_trials': 'Finished Trials',
                'ignore_rate': 'Ignore Rate',
                'total_trials': 'Total Trials',
                'foraging_performance': 'Foraging Performance',
                'abs(bias_naive)': 'Bias'
            };

            let activeFeatures = [];
            features.forEach(feature => {
                const category = subjectData[feature + '_category'];
                const percentile = subjectData[feature + '_session_percentile'];

                if (category && category !== 'N' && category !== 'NS') {
                    activeFeatures.push({
                        name: featureNames[feature],
                        category: category,
                        percentile: percentile
                    });
                }
            });

            if (activeFeatures.length > 0) {
                html += '<div class="tooltip-features">';
                activeFeatures.forEach(feature => {
                    html += '<div class="tooltip-feature-item">';
                    html += `<span class="tooltip-feature-label">${feature.name}: </span>`;

                    const percentileText = (feature.percentile != null && !isNaN(feature.percentile))
                        ? feature.percentile.toFixed(1) + '%'
                        : 'N/A';
                    const color = getAlertColor(feature.category);
                    const colorStyle = color ? `color: ${color}; font-weight: 600;` : 'font-weight: 600;';

                    html += `<span class="tooltip-feature-value" style="${colorStyle}">${percentileText}</span>`;
                    html += '</div>';
                });
                html += '</div>';
            }

            // Threshold alerts
            const thresholdAlert = subjectData.threshold_alert;
            if (thresholdAlert === 'T') {
                let thresholdAlerts = [];

                // Check specific threshold types
                const totalSessionsAlert = subjectData.total_sessions_alert || '';
                const stageSessionsAlert = subjectData.stage_sessions_alert || '';
                const waterDayAlert = subjectData.water_day_total_alert || '';

                if (totalSessionsAlert.includes('T |')) {
                    const value = totalSessionsAlert.split('|')[1]?.trim();
                    thresholdAlerts.push({
                        type: 'Total Sessions',
                        value: `${value} sessions`
                    });
                }

                if (stageSessionsAlert.includes('T |')) {
                    const parts = stageSessionsAlert.split('|');
                    const stage = parts[1]?.trim();
                    const sessions = parts[2]?.trim();
                    thresholdAlerts.push({
                        type: 'Stage Sessions',
                        value: `${stage}: ${sessions}`
                    });
                }

                if (waterDayAlert.includes('T |')) {
                    const value = waterDayAlert.split('|')[1]?.trim();
                    thresholdAlerts.push({
                        type: 'Water Day Total',
                        value: `${value} mL`
                    });
                }

                if (thresholdAlerts.length > 0) {
                    html += '<div class="tooltip-thresholds">';
                    thresholdAlerts.forEach(alert => {
                        html += '<div class="tooltip-threshold-item">';
                        html += `<span class="tooltip-threshold-label">${alert.type}: </span>`;
                        html += `<span class="tooltip-threshold-value" style="color: #795548; font-weight: 600;">${alert.value}</span>`;
                        html += '</div>';
                    });
                    html += '</div>';
                }
            }

            html += '</div>';
            return html;
        }

        // Function to show tooltip
        function showTooltip(subjectId, mouseX, mouseY) {
            if (!tooltip) return;

            // Clear any hide timeout
            if (hideTimeout) {
                clearTimeout(hideTimeout);
                hideTimeout = null;
            }

            // Find subject data in table
            const subjectData = table_data.find(row => row.subject_id === subjectId);
            if (!subjectData) return;

            // Create tooltip content
            const content = createTooltipContent(subjectData);
            tooltip.innerHTML = content;

            // Position tooltip near cursor but avoid edges
            const viewportWidth = window.innerWidth;
            const viewportHeight = window.innerHeight;
            const tooltipWidth = 220;
            const tooltipHeight = 150;

            let left = mouseX + 10;
            let top = mouseY - 10;

            if (left + tooltipWidth > viewportWidth) {
                left = mouseX - tooltipWidth - 10;
            }
            if (top + tooltipHeight > viewportHeight) {
                top = mouseY - tooltipHeight - 10;
            }
            if (left < 0) left = 10;
            if (top < 0) top = 10;

            // Set position and show
            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';
            tooltip.style.opacity = '1';
            tooltip.classList.remove('hidden');

            currentHoveredSubject = subjectId;
        }

        // Function to hide tooltip
        function hideTooltip() {
            if (!tooltip) return;

            hideTimeout = setTimeout(() => {
                tooltip.style.opacity = '0';
                tooltip.classList.add('hidden');
                currentHoveredSubject = null;
            }, 100);
        }

        // Setup hover listeners
        function setupHoverListeners() {
            const cells = table.querySelectorAll('td[data-dash-column="subject_id"]');

            cells.forEach(cell => {
                // Remove existing listeners
                if (cell._tooltipMouseEnter) cell.removeEventListener('mouseenter', cell._tooltipMouseEnter);
                if (cell._tooltipMouseLeave) cell.removeEventListener('mouseleave', cell._tooltipMouseLeave);
                if (cell._tooltipMouseMove) cell.removeEventListener('mousemove', cell._tooltipMouseMove);

                // Add new listeners
                cell._tooltipMouseEnter = function(e) {
                    const subjectId = this.textContent.trim();
                    if (subjectId && subjectId !== currentHoveredSubject) {
                        showTooltip(subjectId, e.clientX, e.clientY);
                    }
                };

                cell._tooltipMouseLeave = function(e) {
                    hideTooltip();
                };

                cell._tooltipMouseMove = function(e) {
                    const subjectId = this.textContent.trim();
                    if (subjectId === currentHoveredSubject && tooltip.style.opacity === '1') {
                        // Update position to follow cursor
                        const viewportWidth = window.innerWidth;
                        const viewportHeight = window.innerHeight;
                        const tooltipWidth = 220;
                        const tooltipHeight = 150;

                        let left = e.clientX + 10;
                        let top = e.clientY - 10;

                        if (left + tooltipWidth > viewportWidth) {
                            left = e.clientX - tooltipWidth - 10;
                        }
                        if (top + tooltipHeight > viewportHeight) {
                            top = e.clientY - tooltipHeight - 10;
                        }
                        if (left < 0) left = 10;
                        if (top < 0) top = 10;

                        tooltip.style.left = left + 'px';
                        tooltip.style.top = top + 'px';
                    }
                };

                cell.addEventListener('mouseenter', cell._tooltipMouseEnter);
                cell.addEventListener('mouseleave', cell._tooltipMouseLeave);
                cell.addEventListener('mousemove', cell._tooltipMouseMove);
            });
        }

        // Setup listeners when table data changes
        setupHoverListeners();

        console.log('Tooltip setup complete');
        return window.dash_clientside.no_update;
    }
    """,
    Output("tooltip-setup-complete", "data"),
    [
        Input("session-table", "data"),
        Input("session-table", "id"),
    ],  # This acts as a trigger for setup
)
