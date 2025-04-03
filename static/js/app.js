document.addEventListener('DOMContentLoaded', function() {
    // Form elements
    const uploadForm = document.getElementById('upload-form');
    const queryForm = document.getElementById('query-form');
    const uploadStatus = document.getElementById('upload-status');
    const uploadResult = document.getElementById('upload-result');
    const queryStatus = document.getElementById('query-status');
    const responseContainer = document.getElementById('response-container');
    const responseContent = document.getElementById('response-content');
    const sourcesContainer = document.getElementById('sources-container');
    const sourcesList = document.getElementById('sources-list');

    // Handle document upload
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading status
        uploadStatus.classList.remove('d-none');
        uploadResult.classList.add('d-none');
        
        const formData = new FormData(uploadForm);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            // Hide loading status
            uploadStatus.classList.add('d-none');
            uploadResult.classList.remove('d-none');
            
            if (response.ok) {
                uploadResult.innerHTML = `
                    <div class="alert alert-success" role="alert">
                        <h5 class="alert-heading"><i class="bi bi-check-circle-fill me-2"></i>Upload Successful</h5>
                        <p>${data.message}</p>
                        <hr>
                        <p class="mb-0">Processed ${data.document_size} characters into ${data.chunks} chunks.</p>
                    </div>
                `;
            } else {
                uploadResult.innerHTML = `
                    <div class="alert alert-danger" role="alert">
                        <h5 class="alert-heading"><i class="bi bi-exclamation-triangle-fill me-2"></i>Upload Failed</h5>
                        <p>${data.error}</p>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Error:', error);
            uploadStatus.classList.add('d-none');
            uploadResult.classList.remove('d-none');
            uploadResult.innerHTML = `
                <div class="alert alert-danger" role="alert">
                    <h5 class="alert-heading"><i class="bi bi-exclamation-triangle-fill me-2"></i>Error</h5>
                    <p>An unexpected error occurred while processing your document.</p>
                </div>
            `;
        }
    });
    
    // Handle query submission
    queryForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const query = document.getElementById('query').value.trim();
        
        if (!query) {
            alert('Please enter a question');
            return;
        }
        
        // Get selected RAG mode
        const ragModeElement = document.querySelector('input[name="rag-mode"]:checked');
        const ragMode = ragModeElement ? ragModeElement.value : 'self';
        
        // Update status message based on mode
        const statusMessageEl = document.getElementById('query-status-message');
        if (ragMode === 'agent') {
            statusMessageEl.innerHTML = 'Running Agentic RAG with iterative improvement... <br><small class="text-muted">This may take a bit longer than Self-RAG</small>';
        } else {
            statusMessageEl.textContent = 'Generating response...';
        }
        
        // Show loading status
        queryStatus.classList.remove('d-none');
        responseContainer.classList.add('d-none');
        sourcesContainer.classList.add('d-none');
        
        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    query: query,
                    rag_mode: ragMode 
                })
            });
            
            const data = await response.json();
            
            // Hide loading status
            queryStatus.classList.add('d-none');
            
            if (response.ok) {
                // Display response
                responseContainer.classList.remove('d-none');
                responseContent.innerHTML = formatResponse(data.response);
                
                // Display sources if available
                if (data.sources && data.sources.length > 0) {
                    sourcesContainer.classList.remove('d-none');
                    sourcesList.innerHTML = '';
                    
                    // Check RAG mode and display appropriate metrics
                    if (data.rag_mode === 'agent' && data.agent_rag_metrics) {
                        // Display Agentic RAG metrics
                        const metricsEl = document.createElement('div');
                        metricsEl.className = 'card bg-dark mb-3';
                        
                        const qualityScore = data.agent_rag_metrics.context_quality || '?';
                        const iterations = data.agent_rag_metrics.iterations || 0;
                        const followUpQueries = data.agent_rag_metrics.follow_up_queries || [];
                        const missingInfo = data.agent_rag_metrics.missing_info || [];
                        
                        // Quality rating
                        const qualityRating = parseInt(qualityScore) >= 7 ? 'High' : 
                                             (parseInt(qualityScore) >= 4 ? 'Medium' : 'Low');
                        const qualityClass = parseInt(qualityScore) >= 7 ? 'text-success' : 
                                           (parseInt(qualityScore) >= 4 ? 'text-warning' : 'text-danger');
                        
                        // Build metrics HTML
                        let metricsHTML = `
                            <div class="card-body">
                                <h6 class="card-subtitle mb-2 text-muted">Agentic RAG Analysis</h6>
                                <div class="small mb-2">
                                    <span class="badge bg-info me-2">Initial chunks: ${data.agent_rag_metrics.initial_chunks}</span>
                                    <span class="badge bg-success me-2">Used chunks: ${data.agent_rag_metrics.used_chunks}</span>
                                    <span class="badge bg-warning me-2">Filtered out: ${data.agent_rag_metrics.filtered_out}</span>
                                    <span class="badge bg-primary me-2">Iterations: ${iterations}</span>
                                </div>
                                <div class="alert alert-info py-2 px-3 mb-2 small">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <span>
                                            <i class="bi bi-graph-up me-1"></i>
                                            Context Quality: <span class="${qualityClass}">${qualityScore}/10 (${qualityRating})</span>
                                        </span>
                                    </div>
                                </div>`;
                                
                        // Add follow-up queries if they exist
                        if (followUpQueries && followUpQueries.length > 0) {
                            metricsHTML += `
                                <div class="mt-2 small">
                                    <strong><i class="bi bi-search me-1"></i>Follow-up Queries:</strong>
                                    <ul class="mb-0 ps-3">
                                        ${followUpQueries.map(q => `<li>${q}</li>`).join('')}
                                    </ul>
                                </div>`;
                        }
                        
                        // Add missing information if it exists
                        if (missingInfo && missingInfo.length > 0) {
                            metricsHTML += `
                                <div class="mt-2 small">
                                    <strong><i class="bi bi-exclamation-triangle me-1"></i>Missing Information:</strong>
                                    <ul class="mb-0 ps-3">
                                        ${Array.isArray(missingInfo) ? missingInfo.map(info => `<li>${info}</li>`).join('') : `<li>${missingInfo}</li>`}
                                    </ul>
                                </div>`;
                        }
                        
                        metricsHTML += `</div>`;
                        metricsEl.innerHTML = metricsHTML;
                        sourcesList.appendChild(metricsEl);
                        
                    } else if (data.self_rag_metrics) {
                        // Display Self-RAG metrics
                        const metricsEl = document.createElement('div');
                        metricsEl.className = 'card bg-dark mb-3';
                        
                        const sufficient = data.self_rag_metrics.is_sufficient;
                        const missingInfo = data.self_rag_metrics.missing_info;
                        
                        metricsEl.innerHTML = `
                            <div class="card-body">
                                <h6 class="card-subtitle mb-2 text-muted">Self-RAG Analysis</h6>
                                <div class="small mb-2">
                                    <span class="badge bg-info me-2">Initial chunks: ${data.self_rag_metrics.initial_chunks}</span>
                                    <span class="badge bg-success me-2">Relevant chunks: ${data.self_rag_metrics.filtered_chunks}</span>
                                    <span class="badge bg-warning me-2">Filtered out: ${data.self_rag_metrics.filtered_out}</span>
                                </div>
                                <div class="alert ${sufficient ? 'alert-success' : 'alert-warning'} py-2 px-3 mb-0 small">
                                    <i class="bi ${sufficient ? 'bi-check-circle' : 'bi-exclamation-triangle'} me-2"></i>
                                    ${sufficient ? 'Context is sufficient to answer the query' : 'Some information may be missing: ' + missingInfo}
                                </div>
                            </div>
                        `;
                        
                        sourcesList.appendChild(metricsEl);
                    }
                    
                    // Display sources with relevance indicators
                    data.sources.forEach((source) => {
                        const sourceItem = document.createElement('div');
                        sourceItem.className = `source-item ${source.relevant ? 'source-relevant' : 'source-filtered'}`;
                        
                        // Determine badge style based on relevance
                        const badgeClass = source.relevant ? 'bg-success' : 'bg-secondary';
                        const relevanceIcon = source.relevant ? 
                            '<i class="bi bi-check-circle-fill me-1"></i>' : 
                            '<i class="bi bi-dash-circle me-1"></i>';
                            
                        sourceItem.innerHTML = `
                            <div class="d-flex align-items-start">
                                <span class="badge ${badgeClass} me-2">
                                    ${relevanceIcon} Source ${source.index}
                                </span>
                                <div>${source.text}</div>
                            </div>
                        `;
                        sourcesList.appendChild(sourceItem);
                    });
                } else {
                    sourcesContainer.classList.add('d-none');
                }
            } else {
                responseContainer.classList.remove('d-none');
                responseContent.innerHTML = `
                    <div class="alert alert-danger" role="alert">
                        <h5 class="alert-heading"><i class="bi bi-exclamation-triangle-fill me-2"></i>Error</h5>
                        <p>${data.error}</p>
                    </div>
                `;
                sourcesContainer.classList.add('d-none');
            }
        } catch (error) {
            console.error('Error:', error);
            queryStatus.classList.add('d-none');
            responseContainer.classList.remove('d-none');
            responseContent.innerHTML = `
                <div class="alert alert-danger" role="alert">
                    <h5 class="alert-heading"><i class="bi bi-exclamation-triangle-fill me-2"></i>Error</h5>
                    <p>An unexpected error occurred while processing your question.</p>
                </div>
            `;
            sourcesContainer.classList.add('d-none');
        }
    });
    
    // Format response with markdown-like syntax
    function formatResponse(text) {
        if (!text) return '';
        
        // Replace newlines with <br>
        let formatted = text.replace(/\n/g, '<br>');
        
        // Simple markdown-like formatting
        // Bold
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Italics
        formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Headers
        formatted = formatted.replace(/#{3} (.*?)(?:\n|$)/g, '<h5>$1</h5>');
        formatted = formatted.replace(/#{2} (.*?)(?:\n|$)/g, '<h4>$1</h4>');
        formatted = formatted.replace(/# (.*?)(?:\n|$)/g, '<h3>$1</h3>');
        
        return formatted;
    }
});
