from dash import ALL, Input, Output, clientside_callback

# Track scroll position and detect which session card is most visible
clientside_callback(
    """
    function(dummy, n_intervals) {
        // Only run if the subject detail page is visible
        const subjectDetailPage = document.getElementById('subject-detail-page');
        if (!subjectDetailPage || subjectDetailPage.style.display === 'none') {
            return {visible_session: null};
        }

        const scrollContainer = document.getElementById('session-list-scroll-container');
        if (!scrollContainer) return {visible_session: null};

        // Check if the scroll container is actually visible in the viewport
        const containerRect = scrollContainer.getBoundingClientRect();
        if (containerRect.height === 0 || containerRect.width === 0) {
            return {visible_session: null};
        }

        // Function to throttle scroll events - increased throttle time
        function throttle(func, limit) {
            let inThrottle;
            return function() {
                const args = arguments;
                const context = this;
                if (!inThrottle) {
                    func.apply(context, args);
                    inThrottle = true;
                    setTimeout(() => inThrottle = false, limit);
                }
            }
        }

        // Function to determine which session card is most visible (center-based approach)
        function getMostVisibleSession() {
            // Use data-* attributes instead of complex JSON id selector
            const cards = document.querySelectorAll('[data-debug^="session-card-"]');
            if (!cards.length) return null;

            const containerRect = scrollContainer.getBoundingClientRect();
            const containerTop = containerRect.top;
            const containerBottom = containerRect.bottom;
            const containerCenter = (containerTop + containerBottom) / 2;

            let minDistanceToCenter = Infinity;
            let mostVisibleCard = null;

            cards.forEach(card => {
                const cardRect = card.getBoundingClientRect();

                // Only consider cards that are at least partially visible
                const visibleTop = Math.max(cardRect.top, containerTop);
                const visibleBottom = Math.min(cardRect.bottom, containerBottom);

                if (visibleBottom > visibleTop) {
                    // Calculate the center of the visible portion of the card
                    const visibleCenter = (visibleTop + visibleBottom) / 2;

                    // Calculate distance from visible card center to container center
                    const distanceToCenter = Math.abs(visibleCenter - containerCenter);

                    // Prefer the card whose visible center is closest to the container center
                    if (distanceToCenter < minDistanceToCenter) {
                        minDistanceToCenter = distanceToCenter;
                        mostVisibleCard = card;
                    }
                }
            });

            if (mostVisibleCard) {
                // Extract subject and session from data attributes
                const subjectId = mostVisibleCard.getAttribute('data-subject-id');
                const sessionNum = mostVisibleCard.getAttribute('data-session-num');

                if (subjectId && sessionNum) {
                    return `${subjectId}-${sessionNum}`;
                }

                // Fallback to parsing ID if data attributes not available
                try {
                    const idStr = mostVisibleCard.id;
                    const idJson = JSON.parse(idStr);
                    return idJson.index;
                } catch (e) {
                    console.error("Error extracting session info:", e);
                    return null;
                }
            }

            return null;
        }

        // Initialize session tracking if not already done
        if (!window.sessionScrollTracking) {
            window.sessionScrollTracking = {
                lastSession: null,
                throttledUpdate: throttle(function() {
                    // Only update if the subject detail page is still visible
                    const subjectDetailPage = document.getElementById('subject-detail-page');
                    if (subjectDetailPage && subjectDetailPage.style.display !== 'none') {
                        window.sessionScrollTracking.lastSession = getMostVisibleSession();
                        window.sessionScrollTracking.needsUpdate = true;
                    }
                }, 150), // Increased throttle from 50ms to 150ms
                needsUpdate: true,
                listenerAttached: false
            };
        }

        // Only attach scroll listener if not already attached and container exists
        if (!window.sessionScrollTracking.listenerAttached && scrollContainer) {
            // Use passive event listener to improve performance and prevent interference
            scrollContainer.addEventListener('scroll', window.sessionScrollTracking.throttledUpdate, { passive: true });
            window.sessionScrollTracking.listenerAttached = true;
        }

        // Reduce update frequency - only check every 3rd interval (300ms instead of 100ms)
        if (n_intervals % 3 !== 0) {
            return {visible_session: window.sessionScrollTracking.lastSession};
        }

        // Only get the current visible session and trigger updates when needed
        if (window.sessionScrollTracking.needsUpdate) {
            const currentVisibleSession = getMostVisibleSession();
            window.sessionScrollTracking.lastSession = currentVisibleSession;
            window.sessionScrollTracking.needsUpdate = false;
            return {visible_session: currentVisibleSession};
        }

        // Return the last known visible session without recalculating
        return {visible_session: window.sessionScrollTracking.lastSession};
    }
    """,
    Output("session-scroll-state", "data"),
    [
        Input("session-list-container", "children"),
        Input("scroll-tracker-interval", "n_intervals"),
    ],
)

# Combined callback to handle session card highlighting from both clicks and scrolling
clientside_callback(
    """
    function(n_clicks_list, visible_session_data) {
        // Check if this was triggered by a click
        let triggeredByClick = false;
        let clickedIdx = null;

        for (let i = 0; i < n_clicks_list.length; i++) {
            if (n_clicks_list[i] && n_clicks_list[i] > 0) {
                triggeredByClick = true;
                clickedIdx = i;
                break;
            }
        }

        // Get all cards using data attribute instead of complex JSON id
        const cards = document.querySelectorAll('[data-debug^="session-card-"]');

        // Default all cards to inactive
        let classes = Array(n_clicks_list.length).fill("session-card");

        if (triggeredByClick && clickedIdx !== null) {
            // If triggered by a click, only highlight the clicked card
            classes[clickedIdx] = "session-card active";
        } else {
            // Otherwise, check the scroll position
            const visibleSession = visible_session_data && visible_session_data.visible_session;

            if (visibleSession) {
                // For each card, determine if it matches the visible session
                cards.forEach((card, index) => {
                    // Skip if index is out of bounds for our classes array
                    if (index >= classes.length) return;

                    // Get the session info from data attributes
                    const subjectId = card.getAttribute('data-subject-id');
                    const sessionNum = card.getAttribute('data-session-num');
                    const cardIndex = `${subjectId}-${sessionNum}`;

                    if (cardIndex === visibleSession) {
                        classes[index] = "session-card active";
                    }
                });
            }
        }

        return classes;
    }
    """,
    Output({"type": "session-card", "index": ALL}, "className"),
    [
        Input({"type": "session-card", "index": ALL}, "n_clicks"),
        Input("session-scroll-state", "data"),
    ],
)

# Cleanup scroll tracking when subject detail page is hidden
clientside_callback(
    """
    function(page_style) {
        // Clean up scroll tracking when subject detail page is hidden
        if (page_style && page_style.display === 'none' && window.sessionScrollTracking) {
            // Remove the scroll event listener to prevent interference with page scrolling
            const scrollContainer = document.getElementById('session-list-scroll-container');
            if (scrollContainer && window.sessionScrollTracking.listenerAttached) {
                scrollContainer.removeEventListener('scroll', window.sessionScrollTracking.throttledUpdate);
                window.sessionScrollTracking.listenerAttached = false;
                window.sessionScrollTracking.needsUpdate = false;
                window.sessionScrollTracking.lastSession = null;
            }
        }
        return {};  // Return empty object since we're not updating any outputs
    }
    """,
    Output("session-card-selected", "data"),  # Use existing store as dummy output
    [Input("subject-detail-page", "style")],
)
