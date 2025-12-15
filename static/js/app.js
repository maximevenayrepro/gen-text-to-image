// Chart instances (global to persist across updates)
let cpuChart = null;
let ramChart = null;
let gpuChart = null;
let gpuMemChart = null;
let gpuSharedMemChart = null;

// Chart data history (keep last 30 points)
const chartDataHistory = {
    cpu: [],
    ram: [],
    gpu: [],
    gpuMem: [],
    gpuSharedMem: []
};

function initCharts() {
    const cpuCanvas = document.getElementById("cpu-chart");
    const ramCanvas = document.getElementById("ram-chart");
    const gpuCanvas = document.getElementById("gpu-chart");
    const gpuMemCanvas = document.getElementById("gpu-mem-chart");
    const gpuSharedMemCanvas = document.getElementById("gpu-shared-mem-chart");
    
    if (!cpuCanvas || !ramCanvas) return;

    const chartOptions = {
        type: 'line',
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            },
            scales: {
                x: { display: false },
                y: {
                    display: false,
                    min: 0,
                    max: 100
                }
            },
            elements: {
                point: { radius: 0 },
                line: {
                    borderWidth: 2,
                    tension: 0.4
                }
            }
        }
    };

    // Initialize CPU chart
    if (!cpuChart) {
        cpuChart = new Chart(cpuCanvas, {
            ...chartOptions,
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    borderColor: '#60a5fa',
                    backgroundColor: 'rgba(96, 165, 250, 0.1)',
                    fill: true
                }]
            }
        });
    }

    // Initialize RAM chart
    if (!ramChart) {
        ramChart = new Chart(ramCanvas, {
            ...chartOptions,
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    borderColor: '#34d399',
                    backgroundColor: 'rgba(52, 211, 153, 0.1)',
                    fill: true
                }]
            }
        });
    }

    // Initialize GPU chart
    if (gpuCanvas && !gpuChart) {
        gpuChart = new Chart(gpuCanvas, {
            ...chartOptions,
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    fill: true
                }]
            }
        });
    }

    // Initialize GPU Memory chart
    if (gpuMemCanvas && !gpuMemChart) {
        gpuMemChart = new Chart(gpuMemCanvas, {
            ...chartOptions,
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    fill: true
                }]
            }
        });
    }

    // Initialize GPU Shared Memory chart
    if (gpuSharedMemCanvas && !gpuSharedMemChart) {
        gpuSharedMemChart = new Chart(gpuSharedMemCanvas, {
            ...chartOptions,
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    borderColor: '#ec4899',
                    backgroundColor: 'rgba(236, 72, 153, 0.1)',
                    fill: true
                }]
            }
        });
    }
}

function updateCharts(cpuPercent, ramPercent, gpuPercent, gpuMemPercent, gpuSharedMemPercent) {
    // Add new data points
    chartDataHistory.cpu.push(cpuPercent);
    chartDataHistory.ram.push(ramPercent);
    if (gpuPercent !== undefined && gpuPercent !== null) {
        chartDataHistory.gpu.push(gpuPercent);
    }
    if (gpuMemPercent !== undefined && gpuMemPercent !== null) {
        chartDataHistory.gpuMem.push(gpuMemPercent);
    }
    if (gpuSharedMemPercent !== undefined && gpuSharedMemPercent !== null) {
        chartDataHistory.gpuSharedMem.push(gpuSharedMemPercent);
    }

    // Keep only last 30 points
    if (chartDataHistory.cpu.length > 30) {
        chartDataHistory.cpu.shift();
        chartDataHistory.ram.shift();
        if (chartDataHistory.gpu.length > 30) chartDataHistory.gpu.shift();
        if (chartDataHistory.gpuMem.length > 30) chartDataHistory.gpuMem.shift();
        if (chartDataHistory.gpuSharedMem.length > 30) chartDataHistory.gpuSharedMem.shift();
    }

    // Update CPU chart
    if (cpuChart) {
        cpuChart.data.labels = chartDataHistory.cpu.map((_, i) => i);
        cpuChart.data.datasets[0].data = chartDataHistory.cpu;
        cpuChart.update('none'); // 'none' mode for smooth animation
    }

    // Update RAM chart
    if (ramChart) {
        ramChart.data.labels = chartDataHistory.ram.map((_, i) => i);
        ramChart.data.datasets[0].data = chartDataHistory.ram;
        ramChart.update('none');
    }

    // Update GPU chart
    if (gpuChart) {
        gpuChart.data.labels = chartDataHistory.gpu.map((_, i) => i);
        gpuChart.data.datasets[0].data = chartDataHistory.gpu;
        gpuChart.update('none');
    }

    // Update GPU Memory chart
    if (gpuMemChart) {
        gpuMemChart.data.labels = chartDataHistory.gpuMem.map((_, i) => i);
        gpuMemChart.data.datasets[0].data = chartDataHistory.gpuMem;
        gpuMemChart.update('none');
    }

    // Update GPU Shared Memory chart
    if (gpuSharedMemChart) {
        gpuSharedMemChart.data.labels = chartDataHistory.gpuSharedMem.map((_, i) => i);
        gpuSharedMemChart.data.datasets[0].data = chartDataHistory.gpuSharedMem;
        gpuSharedMemChart.update('none');
    }
}

async function fetchSystemStats() {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/05956735-5a7a-43dd-9f81-66a8a9b73d4e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:fetchSystemStats',message:'fetchSystemStats called',data:{timestamp:Date.now()},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'H1'})}).catch(()=>{});
    // #endregion
    try {
        const response = await fetch("/api/system_stats");
        if (!response.ok) {
            throw new Error("Network response was not ok");
        }
        const data = await response.json();
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/05956735-5a7a-43dd-9f81-66a8a9b73d4e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:fetchSystemStats',message:'Stats data received',data:{hasData:!!data,hasCpuRam:!!data?.cpu_ram,hasGpus:!!data?.gpus},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'H1'})}).catch(()=>{});
        // #endregion
        renderSystemStats(data);
    } catch (e) {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/05956735-5a7a-43dd-9f81-66a8a9b73d4e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:fetchSystemStats',message:'fetchSystemStats error',data:{error:e.message},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'H1'})}).catch(()=>{});
        // #endregion
        const container = document.getElementById("system-stats");
        if (container) {
            container.innerText = "Failed to load system stats.";
        }
    }
}

function renderSystemStats(data) {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/05956735-5a7a-43dd-9f81-66a8a9b73d4e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:renderSystemStats',message:'renderSystemStats called',data:{hasData:!!data},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'H1'})}).catch(()=>{});
    // #endregion
    const container = document.getElementById("system-stats");
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/05956735-5a7a-43dd-9f81-66a8a9b73d4e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:renderSystemStats',message:'Container check',data:{containerExists:!!container},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'H1'})}).catch(()=>{});
    // #endregion
    if (!container) return;

    if (!data) {
        container.innerText = "No data.";
        return;
    }

    const cpuRam = data.cpu_ram || {};
    const gpus = data.gpus || [];

    // Initialize charts if not already done
    if (!cpuChart || !ramChart || !gpuChart || !gpuMemChart || !gpuSharedMemChart) {
        initCharts();
    }

    // Update metric values (preserve canvas elements)
    const cpuValueEl = document.getElementById("cpu-value");
    const ramValueEl = document.getElementById("ram-value");
    const gpuValueEl = document.getElementById("gpu-value");
    const gpuMemValueEl = document.getElementById("gpu-mem-value");
    const gpuSharedMemValueEl = document.getElementById("gpu-shared-mem-value");
    
    if (cpuValueEl) {
        cpuValueEl.textContent = (cpuRam.cpu_percent ?? "?") + " %";
    }
    if (ramValueEl) {
        ramValueEl.textContent = (cpuRam.ram_percent ?? "?") + " %";
    }

    // Get GPU stats (use first GPU if available)
    let gpuPercent = null;
    let gpuMemPercent = null;
    let gpuSharedMemPercent = null;
    
    if (gpus && gpus.length > 0) {
        const firstGpu = gpus[0];
        gpuPercent = firstGpu.gpu_util_percent ?? null;
        gpuMemPercent = firstGpu.mem_util_percent ?? null;
        gpuSharedMemPercent = firstGpu.shared_mem_percent ?? null;
        
        if (gpuValueEl) {
            gpuValueEl.textContent = (gpuPercent ?? "?") + " %";
        }
        if (gpuMemValueEl) {
            gpuMemValueEl.textContent = (gpuMemPercent ?? "?") + " %";
        }
        if (gpuSharedMemValueEl) {
            gpuSharedMemValueEl.textContent = (gpuSharedMemPercent ?? "?") + " %";
        }
    } else {
        if (gpuValueEl) {
            gpuValueEl.textContent = "--";
        }
        if (gpuMemValueEl) {
            gpuMemValueEl.textContent = "--";
        }
        if (gpuSharedMemValueEl) {
            gpuSharedMemValueEl.textContent = "--";
        }
    }

    // Update charts with new data
    if (cpuRam.cpu_percent !== undefined && cpuRam.cpu_percent !== null) {
        updateCharts(
            cpuRam.cpu_percent, 
            cpuRam.ram_percent ?? 0,
            gpuPercent,
            gpuMemPercent,
            gpuSharedMemPercent
        );
    }

    // #region agent log
    const cpuChartAfter = document.getElementById("cpu-chart");
    const ramChartAfter = document.getElementById("ram-chart");
    fetch('http://127.0.0.1:7242/ingest/05956735-5a7a-43dd-9f81-66a8a9b73d4e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:renderSystemStats',message:'Chart elements check after update',data:{cpuChartExists:!!cpuChartAfter,ramChartExists:!!ramChartAfter,cpuChartInstance:!!cpuChart,ramChartInstance:!!ramChart},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'H1'})}).catch(()=>{});
    // #endregion
}

// Initialize charts on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCharts);
} else {
    initCharts();
}

// Initial fetch + periodic refresh
fetchSystemStats();
setInterval(fetchSystemStats, 3000);

    // ----- Generate button handler -----
    document.addEventListener("DOMContentLoaded", function () {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/05956735-5a7a-43dd-9f81-66a8a9b73d4e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:DOMContentLoaded',message:'DOMContentLoaded fired',data:{timestamp:Date.now()},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'H3'})}).catch(()=>{});
        // #endregion
        const form = document.querySelector("form");
        const generateBtn = document.getElementById("generate-btn");
        const generatingPlaceholder = document.getElementById("generating-placeholder");
        const galleryGrid = document.querySelector(".gallery-grid");
        const emptyState = document.querySelector(".empty-state");

        if (!form || !generateBtn) {
            return;
        }

        form.addEventListener("submit", async function (e) {
            e.preventDefault();

            // Disable and gray out the generate button
            generateBtn.disabled = true;
            generateBtn.style.opacity = "0.5";
            generateBtn.style.cursor = "not-allowed";

            // Hide gallery grid and empty state, show generating placeholder
            if (galleryGrid) {
                galleryGrid.style.display = "none";
            }
            if (emptyState) {
                emptyState.style.display = "none";
            }
            if (generatingPlaceholder) {
                generatingPlaceholder.style.display = "block";
            }

            // Collect form data
            const formData = new FormData(form);
            const data = {
                model_name: formData.get("model_name") || "SDXL",
                custom_model_repo: formData.get("custom_model_repo") || "",
                prompt: formData.get("prompt") || "",
                steps: parseInt(formData.get("steps") || "25"),
                height: parseInt(formData.get("height") || "1024"),
                width: parseInt(formData.get("width") || "1024"),
                num_images: parseInt(formData.get("num_images") || "1")
            };

            try {
                const response = await fetch("/api/generate", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (result.success && result.images && result.images.length > 0) {
                    // Hide generating placeholder
                    if (generatingPlaceholder) {
                        generatingPlaceholder.style.display = "none";
                    }

                    // Show gallery grid
                    if (galleryGrid) {
                        galleryGrid.style.display = "grid";
                    }

                    // Clear existing images
                    galleryGrid.innerHTML = "";

                    // Add new images
                    result.images.forEach(img => {
                        const imageCard = document.createElement("div");
                        imageCard.className = "image-card";
                        imageCard.innerHTML = `
                            <div class="image-wrapper">
                                <img src="${img.data_url}" alt="Generated image">
                                <div class="image-overlay"></div>
                                <div class="image-actions">
                                    <form method="post" action="/generate" style="margin:0;">
                                        <input type="hidden" name="model_name" value="${img.model_name}">
                                        <input type="hidden" name="custom_model_repo" value="">
                                        <input type="hidden" name="prompt" value="${img.prompt}">
                                        <input type="hidden" name="steps" value="${img.steps}">
                                        <input type="hidden" name="height" value="${img.height}">
                                        <input type="hidden" name="width" value="${img.width}">
                                        <input type="hidden" name="num_images" value="1">
                                        <button type="submit" class="icon-button" title="Regenerate">
                                            <i class="fa-solid fa-rotate-right"></i>
                                        </button>
                                    </form>
                                    <a href="${img.view_url}" target="_blank" class="icon-link" title="Open in new tab">
                                        <i class="fa-solid fa-up-right-from-square"></i>
                                    </a>
                                    <a href="${img.download_url}" class="icon-link" title="Download">
                                        <i class="fa-solid fa-download"></i>
                                    </a>
                                </div>
                            </div>
                        `;
                        galleryGrid.appendChild(imageCard);
                    });

                    // Hide empty state
                    if (emptyState) {
                        emptyState.style.display = "none";
                    }
                } else {
                    // Show error
                    alert(result.error || "Error generating images");
                    
                    // Show empty state if no images
                    if (galleryGrid && galleryGrid.children.length === 0) {
                        if (emptyState) {
                            emptyState.style.display = "block";
                        }
                    }
                }
            } catch (error) {
                console.error("Error generating images:", error);
                alert("Error generating images: " + error.message);
                
                // Show empty state if no images
                if (galleryGrid && galleryGrid.children.length === 0) {
                    if (emptyState) {
                        emptyState.style.display = "block";
                    }
                }
            } finally {
                // Re-enable the generate button
                generateBtn.disabled = false;
                generateBtn.style.opacity = "1";
                generateBtn.style.cursor = "pointer";
                
                // Hide generating placeholder
                if (generatingPlaceholder) {
                    generatingPlaceholder.style.display = "none";
                }
            }
        });

    // #region agent log
    const hfTokenBtn = document.getElementById("hf-token-btn");
    const addModelBtn = document.getElementById("add-model-btn");
    const randomPromptBtn = document.getElementById("random-prompt-btn");
    const clearPromptBtn = document.getElementById("clear-prompt-btn");
    fetch('http://127.0.0.1:7242/ingest/05956735-5a7a-43dd-9f81-66a8a9b73d4e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:DOMContentLoaded',message:'Button elements check',data:{hfTokenBtnExists:!!hfTokenBtn,addModelBtnExists:!!addModelBtn,randomPromptBtnExists:!!randomPromptBtn,clearPromptBtnExists:!!clearPromptBtn},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'H3'})}).catch(()=>{});
    // #endregion

    // ----- Update HF Token Status Icon -----
    function updateHfTokenStatusIcon() {
        const statusIcon = document.getElementById("hf-token-status-icon");
        if (!statusIcon) return;
        
        fetch("/api/hf_token")
            .then(response => response.json())
            .then(data => {
                if (data.has_token) {
                    statusIcon.className = "fa-solid fa-circle-check";
                    statusIcon.style.color = "#22c55e";
                } else {
                    statusIcon.className = "fa-solid fa-circle-xmark";
                    statusIcon.style.color = "#ef4444";
                }
            })
            .catch(err => {
                console.error("Error checking token status:", err);
                statusIcon.className = "fa-solid fa-circle-xmark";
                statusIcon.style.color = "#ef4444";
            });
    }

    // ----- HF Token Button -----
    if (hfTokenBtn) {
        // Update status icon on page load
        updateHfTokenStatusIcon();
        
        hfTokenBtn.addEventListener("click", function () {
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/05956735-5a7a-43dd-9f81-66a8a9b73d4e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:hfTokenBtn',message:'hf-token-btn clicked',data:{timestamp:Date.now()},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'H3'})}).catch(()=>{});
            // #endregion
            const modal = document.getElementById("hf-token-modal");
            if (modal) {
                modal.style.display = "flex";
                
                // Load saved token from server if exists
                const tokenInput = document.getElementById("hf-token-input");
                if (tokenInput) {
                    tokenInput.value = ""; // Clear input first
                    fetch("/api/hf_token")
                        .then(response => response.json())
                        .then(data => {
                            if (data.has_token && tokenInput) {
                                // Show masked token as placeholder (user can still enter new one)
                                tokenInput.placeholder = `Current: ${data.token}`;
                            }
                        })
                        .catch(err => console.error("Error loading token:", err));
                }
            }
        });
    }

    // ----- Add Model Button -----
    if (addModelBtn) {
        addModelBtn.addEventListener("click", function () {
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/05956735-5a7a-43dd-9f81-66a8a9b73d4e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:addModelBtn',message:'add-model-btn clicked',data:{timestamp:Date.now()},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'H3'})}).catch(()=>{});
            // #endregion
            const modal = document.getElementById("add-model-modal");
            if (modal) {
                modal.style.display = "flex";
                const input = document.getElementById("modal-model-url");
                if (input) {
                    input.value = "";
                    input.focus();
                }
            }
        });
    }

    // ----- Random Prompt Button -----
    if (randomPromptBtn) {
        randomPromptBtn.addEventListener("click", async function () {
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/05956735-5a7a-43dd-9f81-66a8a9b73d4e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:randomPromptBtn',message:'random-prompt-btn clicked',data:{timestamp:Date.now()},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'H3'})}).catch(()=>{});
            // #endregion
            const promptInput = document.getElementById("prompt");
            if (promptInput) {
                try {
                    // Load prompts from API
                    const response = await fetch("/api/prompts");
                    const data = await response.json();
                    
                    if (data.success && data.prompts && data.prompts.length > 0) {
                        const randomIndex = Math.floor(Math.random() * data.prompts.length);
                        promptInput.value = data.prompts[randomIndex];
                    } else {
                        // Fallback to default prompts if API fails
                        const fallbackPrompts = [
                            "A serene landscape at sunset with mountains in the background",
                            "A futuristic cityscape with flying vehicles and neon lights",
                            "A cozy coffee shop interior with warm lighting"
                        ];
                        const randomIndex = Math.floor(Math.random() * fallbackPrompts.length);
                        promptInput.value = fallbackPrompts[randomIndex];
                    }
                } catch (error) {
                    console.error("Error loading prompts:", error);
                    // Fallback to default prompts on error
                    const fallbackPrompts = [
                        "A serene landscape at sunset with mountains in the background",
                        "A futuristic cityscape with flying vehicles and neon lights",
                        "A cozy coffee shop interior with warm lighting"
                    ];
                    const randomIndex = Math.floor(Math.random() * fallbackPrompts.length);
                    promptInput.value = fallbackPrompts[randomIndex];
                }
            }
        });
    }

    // ----- Clear Prompt Button -----
    if (clearPromptBtn) {
        clearPromptBtn.addEventListener("click", function () {
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/05956735-5a7a-43dd-9f81-66a8a9b73d4e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:clearPromptBtn',message:'clear-prompt-btn clicked',data:{timestamp:Date.now()},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'H3'})}).catch(()=>{});
            // #endregion
            const promptInput = document.getElementById("prompt");
            if (promptInput) {
                promptInput.value = "";
            }
        });
    }

    // ----- Add Model Modal -----
    const addModelModal = document.getElementById("add-model-modal");
    const modalCloseBtn = document.getElementById("modal-close-btn");
    const modalCancelBtn = document.getElementById("modal-cancel-btn");
    const modalOkBtn = document.getElementById("modal-ok-btn");

    function closeAddModelModal() {
        if (addModelModal) {
            addModelModal.style.display = "none";
        }
    }

    if (modalCloseBtn) {
        modalCloseBtn.addEventListener("click", closeAddModelModal);
    }
    if (modalCancelBtn) {
        modalCancelBtn.addEventListener("click", closeAddModelModal);
    }
    if (modalOkBtn) {
        modalOkBtn.addEventListener("click", function () {
            const input = document.getElementById("modal-model-url");
            if (input && input.value.trim()) {
                const repoUrl = input.value.trim();
                
                // Register the model via API (without loading it)
                fetch("/api/register_model", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ repo_or_url: repoUrl })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // Update the model select dropdown
                        const modelSelect = document.getElementById("model_name");
                        if (modelSelect) {
                            // Check if option already exists
                            let optionExists = false;
                            for (let i = 0; i < modelSelect.options.length; i++) {
                                if (modelSelect.options[i].value === data.model_name) {
                                    optionExists = true;
                                    break;
                                }
                            }
                            
                            // Add option if it doesn't exist
                            if (!optionExists) {
                                const option = document.createElement("option");
                                option.value = data.model_name;
                                option.textContent = data.model_name;
                                modelSelect.appendChild(option);
                            }
                            
                            // Select the newly added model
                            modelSelect.value = data.model_name;
                        }
                        
                        closeAddModelModal();
                        input.value = ""; // Clear input for next time
                    } else {
                        alert("Error registering model: " + (data.error || "Unknown error"));
                    }
                })
                .catch(err => {
                    console.error("Error registering model:", err);
                    alert("Error registering model: " + err.message);
                });
            }
        });
    }

    // Close modal when clicking outside
    if (addModelModal) {
        addModelModal.addEventListener("click", function (e) {
            if (e.target === addModelModal) {
                closeAddModelModal();
            }
        });
    }

    // ----- HF Token Modal -----
    const hfTokenModal = document.getElementById("hf-token-modal");
    const hfTokenModalCloseBtn = document.getElementById("hf-token-modal-close-btn");
    const hfTokenModalCancelBtn = document.getElementById("hf-token-modal-cancel-btn");
    const hfTokenModalOkBtn = document.getElementById("hf-token-modal-ok-btn");
    const hfTokenInput = document.getElementById("hf-token-input");
    const hfTokenStatus = document.getElementById("hf-token-status");

    function closeHfTokenModal() {
        if (hfTokenModal) {
            hfTokenModal.style.display = "none";
        }
    }

    if (hfTokenModalCloseBtn) {
        hfTokenModalCloseBtn.addEventListener("click", closeHfTokenModal);
    }
    if (hfTokenModalCancelBtn) {
        hfTokenModalCancelBtn.addEventListener("click", closeHfTokenModal);
    }
    if (hfTokenModalOkBtn) {
        hfTokenModalOkBtn.addEventListener("click", function () {
            if (hfTokenInput && hfTokenInput.value.trim()) {
                const token = hfTokenInput.value.trim();
                
                // Save token to server
                fetch("/api/hf_token", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ token: token })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        if (hfTokenStatus) {
                            hfTokenStatus.textContent = "Token saved successfully!";
                            hfTokenStatus.style.color = "#22c55e";
                        }
                        // Update status icon after saving
                        updateHfTokenStatusIcon();
                        setTimeout(closeHfTokenModal, 1500);
                    } else {
                        if (hfTokenStatus) {
                            hfTokenStatus.textContent = data.error || "Failed to save token";
                            hfTokenStatus.style.color = "#ef4444";
                        }
                    }
                })
                .catch(err => {
                    console.error("Error saving token:", err);
                    if (hfTokenStatus) {
                        hfTokenStatus.textContent = "Error saving token";
                        hfTokenStatus.style.color = "#ef4444";
                    }
                });
            } else {
                if (hfTokenStatus) {
                    hfTokenStatus.textContent = "Please enter a valid token";
                    hfTokenStatus.style.color = "#ef4444";
                }
            }
        });
    }

    // Close modal when clicking outside
    if (hfTokenModal) {
        hfTokenModal.addEventListener("click", function (e) {
            if (e.target === hfTokenModal) {
                closeHfTokenModal();
            }
        });
    }

    // #region agent log
    if (hfTokenBtn) {
        fetch('http://127.0.0.1:7242/ingest/05956735-5a7a-43dd-9f81-66a8a9b73d4e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'app.js:DOMContentLoaded',message:'Event listeners attached',data:{hfTokenBtnHasListener:true,addModelBtnHasListener:true,randomPromptBtnHasListener:true,clearPromptBtnHasListener:true},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'H3'})}).catch(()=>{});
    }
    // #endregion
});
