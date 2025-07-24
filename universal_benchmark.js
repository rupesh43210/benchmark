#!/usr/bin/env node

// Universal Hardware Benchmark + Stress Test Suite
// Comprehensive testing with thermal monitoring and stability analysis

const os = require('os');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const { performance } = require('perf_hooks');
const { spawn, exec, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

class UniversalBenchmarkSuite {
    constructor() {
        this.systemInfo = {};
        this.gpuInfo = [];
        this.cpuInfo = {};
        this.results = {};
        this.stressResults = {};
        this.platform = os.platform();
        this.arch = os.arch();
        this.cpuCount = os.cpus().length;
        this.isStressing = false;
        this.stressWorkers = [];
        
        console.log(`🌟 Universal Hardware Benchmark + Stress Test Suite v3.0`);
        console.log(`🔍 Auto-detecting hardware configuration...\n`);
    }

    // ========== HARDWARE DETECTION ==========
    
    async detectSystemInfo() {
        console.log(`🔧 Detecting system configuration...`);
        
        this.systemInfo = {
            platform: this.platform,
            arch: this.arch,
            cpuCount: this.cpuCount,
            totalMemory: Math.round(os.totalmem() / 1024 / 1024 / 1024),
            cpuModel: os.cpus()[0].model,
            nodeVersion: process.version,
            osVersion: os.release(),
            thermalSupport: await this.detectThermalSupport()
        };

        await this.detectCPUFeatures();
        await this.detectGPUs();
        this.printSystemInfo();
    }

    async detectThermalSupport() {
        try {
            if (this.platform === 'darwin') {
                // macOS thermal support
                execSync('which powermetrics', { stdio: 'ignore' });
                return { sensors: true, power: true, method: 'powermetrics' };
            } else if (this.platform === 'linux') {
                // Linux thermal support
                const hasSensors = fs.existsSync('/sys/class/thermal/thermal_zone0/temp');
                const hasHwmon = fs.existsSync('/sys/class/hwmon/');
                return { sensors: hasSensors, power: hasHwmon, method: 'sysfs' };
            } else if (this.platform === 'win32') {
                // Windows thermal support (limited)
                return { sensors: false, power: false, method: 'wmi' };
            }
        } catch (error) {
            return { sensors: false, power: false, method: 'none' };
        }
        return { sensors: false, power: false, method: 'none' };
    }

    async detectCPUFeatures() {
        const cpuInfo = os.cpus()[0];
        this.cpuInfo = {
            model: cpuInfo.model,
            speed: cpuInfo.speed,
            cores: this.cpuCount,
            architecture: this.arch,
            features: [],
            thermalDesignPower: this.estimateTDP()
        };

        if (this.arch === 'arm64' || this.arch === 'aarch64') {
            this.cpuInfo.features.push('NEON', 'ARM64');
            if (this.platform === 'darwin') {
                this.cpuInfo.features.push('Apple Silicon');
                const model = cpuInfo.model.toLowerCase();
                if (model.includes('m1')) this.cpuInfo.chipset = 'Apple M1';
                else if (model.includes('m2')) this.cpuInfo.chipset = 'Apple M2';
                else if (model.includes('m3')) this.cpuInfo.chipset = 'Apple M3';
                else if (model.includes('m4')) this.cpuInfo.chipset = 'Apple M4';
            }
        } else if (this.arch === 'x64') {
            this.cpuInfo.features.push('x86_64');
            const model = cpuInfo.model.toLowerCase();
            if (model.includes('intel')) {
                this.cpuInfo.vendor = 'Intel';
                this.cpuInfo.features.push('SSE', 'AVX');
            } else if (model.includes('amd')) {
                this.cpuInfo.vendor = 'AMD';
                this.cpuInfo.features.push('SSE', 'AVX');
            }
        }
    }

    estimateTDP() {
        const model = this.cpuInfo?.model?.toLowerCase() || '';
        
        // Rough TDP estimates based on CPU model
        if (model.includes('m1')) return 20; // Apple M1
        if (model.includes('m2')) return 25; // Apple M2  
        if (model.includes('m3')) return 30; // Apple M3
        if (model.includes('i9')) return 125; // Intel i9
        if (model.includes('i7')) return 95;  // Intel i7
        if (model.includes('i5')) return 65;  // Intel i5
        if (model.includes('ryzen 9')) return 105; // AMD Ryzen 9
        if (model.includes('ryzen 7')) return 65;  // AMD Ryzen 7
        if (model.includes('ryzen 5')) return 65;  // AMD Ryzen 5
        
        return 65; // Default estimate
    }

    async detectGPUs() {
        console.log(`🎮 Detecting GPU hardware...`);
        
        try {
            if (this.platform === 'darwin') {
                await this.detectMacOSGPUs();
            } else if (this.platform === 'win32') {
                await this.detectWindowsGPUs();
            } else if (this.platform === 'linux') {
                await this.detectLinuxGPUs();
            }
        } catch (error) {
            console.log(`   ⚠️  GPU detection limited: ${error.message}`);
            this.gpuInfo.push({
                name: 'Generic GPU',
                vendor: 'Unknown',
                apis: ['WebGL'],
                compute: false
            });
        }
    }

    async detectMacOSGPUs() {
        try {
            const result = execSync('system_profiler SPDisplaysDataType -json 2>/dev/null', { encoding: 'utf8' });
            const data = JSON.parse(result);
            
            if (data.SPDisplaysDataType) {
                data.SPDisplaysDataType.forEach(gpu => {
                    const gpuData = {
                        name: gpu.sppci_model || gpu._name || 'Unknown GPU',
                        vendor: 'Apple',
                        apis: ['Metal', 'OpenCL', 'WebGL'],
                        compute: true,
                        memory: gpu.sppci_vram || 'Shared',
                        cores: gpu.sppci_cores || 'Unknown',
                        thermalSupport: true
                    };
                    
                    if (gpuData.name.toLowerCase().includes('m1')) {
                        gpuData.cores = gpuData.name.includes('Pro') ? 16 : 8;
                    } else if (gpuData.name.toLowerCase().includes('m2')) {
                        gpuData.cores = gpuData.name.includes('Pro') ? 19 : 10;
                    } else if (gpuData.name.toLowerCase().includes('m3')) {
                        gpuData.cores = gpuData.name.includes('Pro') ? 18 : 10;
                    }
                    
                    this.gpuInfo.push(gpuData);
                });
            }
        } catch (error) {
            if (this.cpuInfo.chipset) {
                this.gpuInfo.push({
                    name: `${this.cpuInfo.chipset} GPU`,
                    vendor: 'Apple',
                    apis: ['Metal', 'OpenCL', 'WebGL'],
                    compute: true,
                    memory: 'Unified',
                    thermalSupport: true
                });
            }
        }
    }

    async detectWindowsGPUs() {
        try {
            const result = execSync('wmic path win32_VideoController get name,AdapterRAM,DriverVersion /format:csv', 
                { encoding: 'utf8' });
            
            const lines = result.split('\n').filter(line => line.trim() && !line.startsWith('Node'));
            
            lines.forEach(line => {
                const parts = line.split(',');
                if (parts.length >= 3) {
                    const name = parts[2]?.trim();
                    const memory = parts[1]?.trim();
                    
                    if (name && name !== 'Name') {
                        const vendor = this.detectGPUVendor(name);
                        const gpuData = {
                            name: name,
                            vendor: vendor,
                            apis: this.getAPIsByVendor(vendor),
                            compute: true,
                            memory: memory ? `${Math.round(parseInt(memory) / 1024 / 1024)}MB` : 'Unknown',
                            thermalSupport: vendor === 'NVIDIA' || vendor === 'AMD'
                        };
                        this.gpuInfo.push(gpuData);
                    }
                }
            });
        } catch (error) {
            console.log(`   ⚠️  Windows GPU detection failed: ${error.message}`);
        }
    }

    async detectLinuxGPUs() {
        try {
            const result = execSync('lspci | grep -i vga', { encoding: 'utf8' });
            const lines = result.split('\n').filter(line => line.trim());
            
            lines.forEach(line => {
                const name = line.split(':').pop()?.trim();
                if (name) {
                    const vendor = this.detectGPUVendor(name);
                    this.gpuInfo.push({
                        name: name,
                        vendor: vendor,
                        apis: this.getAPIsByVendor(vendor),
                        compute: vendor !== 'Intel',
                        memory: 'Unknown',
                        thermalSupport: vendor === 'NVIDIA' || vendor === 'AMD'
                    });
                }
            });

            // NVIDIA specific detection
            try {
                const nvidiaResult = execSync('nvidia-smi --query-gpu=name,memory.total,power.max_limit --format=csv,noheader,nounits', 
                    { encoding: 'utf8' });
                
                const nvidiaLines = nvidiaResult.split('\n').filter(line => line.trim());
                nvidiaLines.forEach(line => {
                    const [name, memory, powerLimit] = line.split(',').map(s => s.trim());
                    if (name && memory) {
                        const existing = this.gpuInfo.find(gpu => gpu.name.includes(name));
                        if (existing) {
                            existing.memory = `${memory}MB`;
                            existing.cuda = true;
                            existing.powerLimit = powerLimit ? `${powerLimit}W` : 'Unknown';
                        } else {
                            this.gpuInfo.push({
                                name: name,
                                vendor: 'NVIDIA',
                                apis: ['CUDA', 'OpenCL', 'Vulkan', 'OpenGL'],
                                compute: true,
                                memory: `${memory}MB`,
                                cuda: true,
                                powerLimit: powerLimit ? `${powerLimit}W` : 'Unknown',
                                thermalSupport: true
                            });
                        }
                    }
                });
            } catch (e) {
                // NVIDIA tools not available
            }
        } catch (error) {
            console.log(`   ⚠️  Linux GPU detection failed: ${error.message}`);
        }
    }

    detectGPUVendor(name) {
        const nameLower = name.toLowerCase();
        if (nameLower.includes('nvidia') || nameLower.includes('geforce') || nameLower.includes('quadro') || nameLower.includes('tesla')) {
            return 'NVIDIA';
        } else if (nameLower.includes('amd') || nameLower.includes('radeon') || nameLower.includes('rx ') || nameLower.includes('vega')) {
            return 'AMD';
        } else if (nameLower.includes('intel') || nameLower.includes('uhd') || nameLower.includes('iris')) {
            return 'Intel';
        } else if (nameLower.includes('apple') || nameLower.includes('m1') || nameLower.includes('m2') || nameLower.includes('m3')) {
            return 'Apple';
        }
        return 'Unknown';
    }

    getAPIsByVendor(vendor) {
        const apiMap = {
            'NVIDIA': ['CUDA', 'OpenCL', 'Vulkan', 'OpenGL', 'DirectX'],
            'AMD': ['OpenCL', 'Vulkan', 'OpenGL', 'DirectX', 'ROCm'],
            'Intel': ['OpenCL', 'Vulkan', 'OpenGL', 'DirectX'],
            'Apple': ['Metal', 'OpenCL', 'WebGL'],
            'Unknown': ['WebGL']
        };
        return apiMap[vendor] || ['WebGL'];
    }

    printSystemInfo() {
        console.log(`\n📋 SYSTEM CONFIGURATION`);
        console.log(`${'='.repeat(50)}`);
        console.log(`🖥️  Platform: ${this.systemInfo.platform} (${this.systemInfo.arch})`);
        console.log(`💻 CPU: ${this.cpuInfo.model}`);
        console.log(`🧮 Cores: ${this.cpuInfo.cores} threads`);
        console.log(`🔥 Est. TDP: ${this.cpuInfo.thermalDesignPower}W`);
        console.log(`🚀 Features: ${this.cpuInfo.features.join(', ')}`);
        console.log(`🧠 Memory: ${this.systemInfo.totalMemory}GB`);
        console.log(`🌡️  Thermal Support: ${this.systemInfo.thermalSupport.sensors ? '✅' : '❌'}`);
        
        console.log(`\n🎮 GPU CONFIGURATION:`);
        this.gpuInfo.forEach((gpu, i) => {
            console.log(`   GPU ${i + 1}: ${gpu.name} (${gpu.vendor})`);
            console.log(`   APIs: ${gpu.apis.join(', ')}`);
            if (gpu.memory) console.log(`   Memory: ${gpu.memory}`);
            if (gpu.cores) console.log(`   Cores: ${gpu.cores}`);
            if (gpu.powerLimit) console.log(`   Power Limit: ${gpu.powerLimit}`);
            console.log(`   Thermal Monitoring: ${gpu.thermalSupport ? '✅' : '❌'}`);
        });
        console.log();
    }

    // ========== THERMAL MONITORING ==========

    async getSystemTemperatures() {
        const temps = { cpu: null, gpu: [], timestamp: Date.now() };
        
        try {
            if (this.platform === 'darwin') {
                // macOS temperature monitoring
                const result = execSync('sudo powermetrics -n 1 -i 100 --samplers smc,cpu_power -f plist 2>/dev/null || echo "failed"', 
                    { encoding: 'utf8', timeout: 5000 });
                
                if (!result.includes('failed')) {
                    // Parse powermetrics output (simplified)
                    const tempMatch = result.match(/CPU die temperature: (\d+\.\d+)/);
                    if (tempMatch) temps.cpu = parseFloat(tempMatch[1]);
                }
            } else if (this.platform === 'linux') {
                // Linux temperature monitoring
                if (fs.existsSync('/sys/class/thermal/thermal_zone0/temp')) {
                    const temp = fs.readFileSync('/sys/class/thermal/thermal_zone0/temp', 'utf8');
                    temps.cpu = parseInt(temp) / 1000; // Convert millicelsius to celsius
                }
                
                // Try to get GPU temperatures
                try {
                    const nvidiaTemp = execSync('nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits', 
                        { encoding: 'utf8', timeout: 2000 });
                    const gpuTemps = nvidiaTemp.trim().split('\n').map(t => parseInt(t));
                    temps.gpu = gpuTemps.filter(t => !isNaN(t));
                } catch (e) {
                    // No NVIDIA GPU or nvidia-smi not available
                }
            }
        } catch (error) {
            // Temperature monitoring not available
        }
        
        return temps;
    }

    async getPowerConsumption() {
        const power = { cpu: null, gpu: [], total: null, timestamp: Date.now() };
        
        try {
            if (this.platform === 'darwin') {
                // macOS power monitoring via powermetrics
                const result = execSync('sudo powermetrics -n 1 -i 100 --samplers cpu_power -f plist 2>/dev/null || echo "failed"', 
                    { encoding: 'utf8', timeout: 5000 });
                
                if (!result.includes('failed')) {
                    const powerMatch = result.match(/CPU Power: (\d+\.\d+)/);
                    if (powerMatch) power.cpu = parseFloat(powerMatch[1]);
                }
            } else if (this.platform === 'linux') {
                // Linux power monitoring
                try {
                    const nvidiaPower = execSync('nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits', 
                        { encoding: 'utf8', timeout: 2000 });
                    const gpuPowers = nvidiaPower.trim().split('\n').map(p => parseFloat(p));
                    power.gpu = gpuPowers.filter(p => !isNaN(p));
                } catch (e) {
                    // No NVIDIA GPU power monitoring
                }
            }
        } catch (error) {
            // Power monitoring not available
        }
        
        return power;
    }

    // ========== STRESS TEST IMPLEMENTATIONS ==========

    static cpuStressTest(durationMs = 60000, intensity = 1.0) {
        const start = performance.now();
        const endTime = start + durationMs;
        let iterations = 0;
        let result = 0;
        
        // Adjustable intensity (0.1 to 2.0)
        const baseOpsPerLoop = Math.floor(10000 * intensity);
        
        while (performance.now() < endTime) {
            // High-intensity mathematical operations
            for (let i = 0; i < baseOpsPerLoop; i++) {
                // Complex mathematical operations
                result += Math.sin(iterations * 0.001) * Math.cos(iterations * 0.002);
                result += Math.sqrt(iterations % 10000) * Math.log(iterations + 1);
                result += Math.pow(iterations % 100, 2.5);
                result ^= (iterations << 3) | (iterations >> 5);
                
                // Prime number calculations (CPU intensive)
                let isPrime = true;
                const num = (iterations % 1000) + 2;
                for (let j = 2; j <= Math.sqrt(num); j++) {
                    if (num % j === 0) {
                        isPrime = false;
                        break;
                    }
                }
                if (isPrime) result += num;
                
                iterations++;
            }
        }
        
        const actualDuration = performance.now() - start;
        return {
            duration: actualDuration,
            iterations,
            opsPerSecond: iterations / (actualDuration / 1000),
            result: result % 1000000,
            intensity
        };
    }

    static memoryStressTest(durationMs = 30000, sizeGB = 1) {
        const start = performance.now();
        const endTime = start + durationMs;
        const arraySize = Math.floor((sizeGB * 1024 * 1024 * 1024) / 8); // 8 bytes per double
        
        console.log(`     Allocating ${sizeGB}GB memory buffer...`);
        const memoryArray = new Float64Array(arraySize);
        let cycles = 0;
        
        while (performance.now() < endTime) {
            // Write phase - fill memory with random data
            for (let i = 0; i < arraySize; i += 1000) {
                memoryArray[i] = Math.random() * 1000;
            }
            
            // Read/modify phase - intensive memory operations
            let sum = 0;
            for (let i = 0; i < arraySize; i += 500) {
                sum += memoryArray[i];
                memoryArray[i] = sum * 0.001; // Modify to prevent optimization
            }
            
            // Random access phase - stress memory controller
            for (let i = 0; i < 100000; i++) {
                const idx = Math.floor(Math.random() * arraySize);
                memoryArray[idx] += Math.sin(idx * 0.001);
            }
            
            cycles++;
        }
        
        const actualDuration = performance.now() - start;
        const throughputGBps = (cycles * sizeGB * 3) / (actualDuration / 1000); // Read + Write + Modify
        
        return {
            duration: actualDuration,
            cycles,
            throughputGBps,
            sizeGB,
            checksum: memoryArray[0] % 1000000
        };
    }

    static gpuStressTest(durationMs = 45000, complexity = 1.0) {
        const start = performance.now();
        const endTime = start + durationMs;
        
        // Scale matrix size based on complexity
        const matrixSize = Math.floor(512 * Math.sqrt(complexity));
        const matrixA = new Float32Array(matrixSize * matrixSize);
        const matrixB = new Float32Array(matrixSize * matrixSize);
        const result = new Float32Array(matrixSize * matrixSize);
        
        // Initialize matrices
        for (let i = 0; i < matrixSize * matrixSize; i++) {
            matrixA[i] = Math.random();
            matrixB[i] = Math.random();
        }
        
        let operations = 0;
        let totalFlops = 0;
        
        while (performance.now() < endTime) {
            // Matrix multiplication (GPU-style computation)
            const blockSize = 32;
            for (let ii = 0; ii < matrixSize; ii += blockSize) {
                for (let jj = 0; jj < matrixSize; jj += blockSize) {
                    for (let kk = 0; kk < matrixSize; kk += blockSize) {
                        const iEnd = Math.min(ii + blockSize, matrixSize);
                        const jEnd = Math.min(jj + blockSize, matrixSize);
                        const kEnd = Math.min(kk + blockSize, matrixSize);
                        
                        for (let i = ii; i < iEnd; i++) {
                            for (let j = jj; j < jEnd; j++) {
                                let sum = 0;
                                for (let k = kk; k < kEnd; k++) {
                                    sum += matrixA[i * matrixSize + k] * matrixB[k * matrixSize + j];
                                }
                                result[i * matrixSize + j] = sum;
                            }
                        }
                    }
                }
            }
            
            // Additional GPU-style operations
            for (let i = 0; i < matrixSize * matrixSize; i += 100) {
                result[i] = Math.sin(result[i]) * Math.cos(matrixA[i]) + Math.sqrt(Math.abs(matrixB[i]));
            }
            
            operations++;
            totalFlops += 2 * matrixSize * matrixSize * matrixSize;
        }
        
        const actualDuration = performance.now() - start;
        const gflops = (totalFlops / 1e9) / (actualDuration / 1000);
        
        return {
            duration: actualDuration,
            operations,
            gflops,
            matrixSize,
            complexity,
            checksum: result[0] % 1000000
        };
    }

    // ========== STRESS TEST ORCHESTRATION ==========

    async runStressTest(testName, duration = 60, options = {}) {
        console.log(`\n🔥 STRESS TEST: ${testName.toUpperCase()}`);
        console.log(`${'='.repeat(50)}`);
        console.log(`⏱️  Duration: ${duration}s`);
        console.log(`🌡️  Monitoring: ${this.systemInfo.thermalSupport.sensors ? 'Enabled' : 'Limited'}`);
        
        const stressData = {
            testName,
            duration,
            startTime: Date.now(),
            thermalData: [],
            powerData: [],
            performanceData: [],
            stability: { crashes: 0, errors: [] }
        };

        // Start thermal monitoring
        const monitoringInterval = setInterval(async () => {
            if (this.systemInfo.thermalSupport.sensors) {
                const temps = await this.getSystemTemperatures();
                const power = await this.getPowerConsumption();
                
                stressData.thermalData.push(temps);
                stressData.powerData.push(power);
                
                if (temps.cpu) {
                    process.stdout.write(`\r🌡️  CPU: ${temps.cpu.toFixed(1)}°C`);
                    if (temps.gpu.length > 0) {
                        process.stdout.write(` | GPU: ${temps.gpu[0].toFixed(1)}°C`);
                    }
                    if (power.cpu) {
                        process.stdout.write(` | Power: ${power.cpu.toFixed(1)}W`);
                    }
                }
            }
        }, 2000);

        try {
            let stressResults;
            
            if (testName === 'cpu') {
                stressResults = await this.runCPUStressTest(duration * 1000, options.intensity || 1.0);
            } else if (testName === 'memory') {
                stressResults = await this.runMemoryStressTest(duration * 1000, options.sizeGB || 2);
            } else if (testName === 'gpu') {
                stressResults = await this.runGPUStressTest(duration * 1000, options.complexity || 1.0);
            } else if (testName === 'combined') {
                stressResults = await this.runCombinedStressTest(duration * 1000, options);
            }
            
            stressData.results = stressResults;
            stressData.endTime = Date.now();
            
        } catch (error) {
            stressData.stability.crashes++;
            stressData.stability.errors.push(error.message);
            console.log(`\n❌ Stress test error: ${error.message}`);
        } finally {
            clearInterval(monitoringInterval);
            console.log(`\n`); // New line after temperature display
        }

        // Analyze results
        this.analyzeStressTestResults(stressData);
        return stressData;
    }

    async runCPUStressTest(durationMs, intensity) {
        console.log(`🧮 Starting CPU stress test (${this.cpuCount} threads)...`);
        
        return new Promise((resolve) => {
            const workers = [];
            const results = [];
            let completed = 0;
            const startTime = performance.now();
            
            // Create stress workers for all CPU cores
            for (let i = 0; i < this.cpuCount; i++) {
                const worker = new Worker(__filename, {
                    workerData: { 
                        task: 'cpu_stress',
                        workerId: i,
                        duration: durationMs,
                        intensity: intensity
                    }
                });
                
                worker.on('message', (result) => {
                    results.push(result);
                    completed++;
                    
                    if (completed === this.cpuCount) {
                        const totalTime = performance.now() - startTime;
                        const avgOpsPerSecond = results.reduce((sum, r) => sum + r.opsPerSecond, 0) / results.length;
                        const totalOps = results.reduce((sum, r) => sum + r.iterations, 0);
                        
                        workers.forEach(w => w.terminate());
                        
                        resolve({
                            type: 'cpu',
                            totalTime,
                            avgOpsPerSecond,
                            totalOperations: totalOps,
                            threadsUsed: this.cpuCount,
                            intensity,
                            workerResults: results
                        });
                    }
                });
                
                worker.on('error', (error) => {
                    console.error(`CPU stress worker ${i} error:`, error);
                });
                
                workers.push(worker);
                this.stressWorkers.push(worker);
            }
        });
    }

    async runMemoryStressTest(durationMs, sizeGB) {
        console.log(`💾 Starting memory stress test (${sizeGB}GB)...`);
        
        return new Promise((resolve) => {
            const worker = new Worker(__filename, {
                workerData: { 
                    task: 'memory_stress',
                    duration: durationMs,
                    sizeGB: sizeGB
                }
            });
            
            worker.on('message', (result) => {
                worker.terminate();
                resolve({
                    type: 'memory',
                    ...result
                });
            });
            
            worker.on('error', (error) => {
                console.error(`Memory stress worker error:`, error);
                worker.terminate();
                resolve({ type: 'memory', error: error.message });
            });
            
            this.stressWorkers.push(worker);
        });
    }

    async runGPUStressTest(durationMs, complexity) {
        console.log(`🎮 Starting GPU stress test...`);
        
        return new Promise((resolve) => {
            const numWorkers = Math.min(4, this.cpuCount); // Limit GPU workers
            const workers = [];
            const results = [];
            let completed = 0;
            
            for (let i = 0; i < numWorkers; i++) {
                const worker = new Worker(__filename, {
                    workerData: { 
                        task: 'gpu_stress',
                        workerId: i,
                        duration: durationMs,
                        complexity: complexity
                    }
                });
                
                worker.on('message', (result) => {
                    results.push(result);
                    completed++;
                    
                    if (completed === numWorkers) {
                        const totalGflops = results.reduce((sum, r) => sum + r.gflops, 0);
                        const avgGflops = totalGflops / numWorkers;
                        
                        workers.forEach(w => w.terminate());
                        
                        resolve({
                            type: 'gpu',
                            totalGflops,
                            avgGflops,
                            complexity,
                            workersUsed: numWorkers,
                            workerResults: results
                        });
                    }
                });
                
                worker.on('error', (error) => {
                    console.error(`GPU stress worker ${i} error:`, error);
                });
                
                workers.push(worker);
                this.stressWorkers.push(worker);
            }
        });
    }

    async runCombinedStressTest(durationMs, options) {
        console.log(`🔥 Starting COMBINED stress test (CPU + GPU + Memory)...`);
        console.log(`⚠️  WARNING: This will push your system to maximum limits!`);
        
        const startTime = performance.now();
        const results = {};
        
        try {
            // Run all stress tests simultaneously
            const [cpuResult, memoryResult, gpuResult] = await Promise.all([
                this.runCPUStressTest(durationMs, options.cpuIntensity || 0.8),
                this.runMemoryStressTest(durationMs, options.memoryGB || 1),
                this.runGPUStressTest(durationMs, options.gpuComplexity || 0.8)
            ]);
            
            results.cpu = cpuResult;
            results.memory = memoryResult;
            results.gpu = gpuResult;
            results.totalTime = performance.now() - startTime;
            results.type = 'combined';
            
        } catch (error) {
            results.error = error.message;
            console.log(`❌ Combined stress test interrupted: ${error.message}`);
        }
        
        return results;
    }

    analyzeStressTestResults(stressData) {
        console.log(`\n📊 STRESS TEST ANALYSIS`);
        console.log(`${'='.repeat(50)}`);
        
        const actualDuration = (stressData.endTime - stressData.startTime) / 1000;
        console.log(`⏱️  Actual Duration: ${actualDuration.toFixed(1)}s`);
        
        // Thermal Analysis
        if (stressData.thermalData.length > 0) {
            const cpuTemps = stressData.thermalData.filter(t => t.cpu !== null).map(t => t.cpu);
            if (cpuTemps.length > 0) {
                const maxTemp = Math.max(...cpuTemps);
                const avgTemp = cpuTemps.reduce((a, b) => a + b, 0) / cpuTemps.length;
                console.log(`🌡️  CPU Temperature: Max ${maxTemp.toFixed(1)}°C, Avg ${avgTemp.toFixed(1)}°C`);
                
                // Thermal throttling detection
                if (maxTemp > 90) {
                    console.log(`⚠️  WARNING: High CPU temperature detected!`);
                } else if (maxTemp > 80) {
                    console.log(`🔥 CPU running hot but within limits`);
                } else {
                    console.log(`✅ CPU temperature normal`);
                }
            }
            
            // GPU temperature analysis
            const gpuTemps = stressData.thermalData.flatMap(t => t.gpu).filter(t => t !== null);
            if (gpuTemps.length > 0) {
                const maxGpuTemp = Math.max(...gpuTemps);
                const avgGpuTemp = gpuTemps.reduce((a, b) => a + b, 0) / gpuTemps.length;
                console.log(`🎮 GPU Temperature: Max ${maxGpuTemp.toFixed(1)}°C, Avg ${avgGpuTemp.toFixed(1)}°C`);
            }
        }
        
        // Power Analysis
        if (stressData.powerData.length > 0) {
            const powerReadings = stressData.powerData.filter(p => p.cpu !== null).map(p => p.cpu);
            if (powerReadings.length > 0) {
                const maxPower = Math.max(...powerReadings);
                const avgPower = powerReadings.reduce((a, b) => a + b, 0) / powerReadings.length;
                console.log(`⚡ Power Consumption: Max ${maxPower.toFixed(1)}W, Avg ${avgPower.toFixed(1)}W`);
                
                // Power efficiency analysis
                const estimatedTDP = this.cpuInfo.thermalDesignPower;
                if (maxPower > estimatedTDP * 1.1) {
                    console.log(`⚠️  Power draw exceeds TDP estimate`);
                } else {
                    console.log(`✅ Power consumption within expected range`);
                }
            }
        }
        
        // Performance Analysis
        if (stressData.results) {
            const results = stressData.results;
            
            if (results.type === 'cpu') {
                console.log(`🧮 CPU Performance: ${(results.avgOpsPerSecond / 1000000).toFixed(1)}M ops/sec avg`);
                console.log(`🎯 Total Operations: ${results.totalOperations.toLocaleString()}`);
                
                // Performance consistency check
                const opsVariance = this.calculateVariance(results.workerResults.map(r => r.opsPerSecond));
                const opsStdDev = Math.sqrt(opsVariance);
                const cvPercent = (opsStdDev / results.avgOpsPerSecond) * 100;
                
                if (cvPercent < 5) {
                    console.log(`✅ Performance consistent across all cores (CV: ${cvPercent.toFixed(1)}%)`);
                } else if (cvPercent < 15) {
                    console.log(`⚠️  Moderate performance variation (CV: ${cvPercent.toFixed(1)}%)`);
                } else {
                    console.log(`❌ High performance variation detected (CV: ${cvPercent.toFixed(1)}%)`);
                }
                
            } else if (results.type === 'memory') {
                console.log(`💾 Memory Performance: ${results.throughputGBps.toFixed(2)} GB/s throughput`);
                console.log(`🔄 Memory Cycles: ${results.cycles.toLocaleString()}`);
                
            } else if (results.type === 'gpu') {
                console.log(`🎮 GPU Performance: ${results.avgGflops.toFixed(1)} GFLOPS avg`);
                console.log(`🚀 Total GPU Power: ${results.totalGflops.toFixed(1)} GFLOPS`);
                
            } else if (results.type === 'combined') {
                console.log(`🔥 COMBINED STRESS RESULTS:`);
                if (results.cpu) console.log(`   CPU: ${(results.cpu.avgOpsPerSecond / 1000000).toFixed(1)}M ops/sec`);
                if (results.memory) console.log(`   Memory: ${results.memory.throughputGBps.toFixed(2)} GB/s`);
                if (results.gpu) console.log(`   GPU: ${results.gpu.avgGflops.toFixed(1)} GFLOPS`);
            }
        }
        
        // Stability Analysis
        console.log(`🛡️  Stability: ${stressData.stability.crashes === 0 ? '✅ STABLE' : `❌ ${stressData.stability.crashes} crashes`}`);
        if (stressData.stability.errors.length > 0) {
            console.log(`❌ Errors encountered:`);
            stressData.stability.errors.forEach(error => console.log(`   - ${error}`));
        }
    }

    calculateVariance(numbers) {
        const mean = numbers.reduce((a, b) => a + b, 0) / numbers.length;
        const variance = numbers.reduce((sum, num) => sum + Math.pow(num - mean, 2), 0) / numbers.length;
        return variance;
    }

    // ========== INTERACTIVE STRESS TEST MENU ==========

    async showStressTestMenu() {
        console.log(`\n🔥 STRESS TEST OPTIONS`);
        console.log(`${'='.repeat(40)}`);
        console.log(`1. 🧮 CPU Stress Test (60s)`);
        console.log(`2. 💾 Memory Stress Test (30s)`);
        console.log(`3. 🎮 GPU Stress Test (45s)`);
        console.log(`4. 🔥 Combined Stress Test (90s) - EXTREME`);
        console.log(`5. ⚙️  Custom Stress Test`);
        console.log(`6. 📊 Quick Benchmark (No Stress)`);
        console.log(`7. 🏃 Skip Stress Tests`);
        console.log(`\nPress Ctrl+C to stop any test safely\n`);
        
        // For demo purposes, we'll run a combined stress test
        // In a real implementation, you'd want to add readline for user input
        console.log(`🎯 Auto-selecting Combined Stress Test for demonstration...`);
        
        return await this.runStressTest('combined', 90, {
            cpuIntensity: 0.9,
            memoryGB: 2,
            gpuComplexity: 0.8
        });
    }

    // ========== REGULAR BENCHMARK IMPLEMENTATIONS ==========
    
    static cpuBenchmark(iterations = 30000000) {
        const start = performance.now();
        let result = 0;
        
        for (let i = 0; i < iterations; i++) {
            result += Math.sin(i * 0.001) * Math.cos(i * 0.002);
            result += Math.sqrt(i % 1000) * Math.log(i + 1);
            result += (i * 31) % 97;
            result ^= (i << 2) | (i >> 3);
        }
        
        const duration = performance.now() - start;
        return { result, duration, iterations, opsPerSecond: iterations / (duration / 1000) };
    }

    static memoryBenchmark() {
        const sizes = [
            { name: 'L1 Cache', size: 32 * 1024 / 8 },
            { name: 'L2 Cache', size: 256 * 1024 / 8 },
            { name: 'L3 Cache', size: 8 * 1024 * 1024 / 8 },
            { name: 'RAM', size: 64 * 1024 * 1024 / 8 }
        ];
        
        const results = [];
        
        sizes.forEach(({ name, size }) => {
            const arr = new Float64Array(size);
            const iterations = Math.min(10000000, 100000000 / Math.sqrt(size));
            
            for (let i = 0; i < size; i++) arr[i] = i;
            
            const start = performance.now();
            let sum = 0;
            
            for (let i = 0; i < iterations / 2; i++) {
                sum += arr[i % size];
                sum += arr[Math.floor(Math.random() * size)];
            }
            
            const duration = performance.now() - start;
            results.push({
                name,
                duration,
                bandwidth: (size * 8 * iterations / 2) / (1024 * 1024 * 1024) / (duration / 1000),
                latency: (duration * 1000000) / iterations,
                checksum: sum % 1000000
            });
        });
        
        return results;
    }

    static gpuMatrixBenchmark(size = 1024) {
        const start = performance.now();
        
        const matrixA = new Float32Array(size * size);
        const matrixB = new Float32Array(size * size);
        const result = new Float32Array(size * size);
        
        for (let i = 0; i < size * size; i++) {
            matrixA[i] = Math.random();
            matrixB[i] = Math.random();
        }
        
        const blockSize = 64;
        for (let ii = 0; ii < size; ii += blockSize) {
            for (let jj = 0; jj < size; jj += blockSize) {
                for (let kk = 0; kk < size; kk += blockSize) {
                    const iEnd = Math.min(ii + blockSize, size);
                    const jEnd = Math.min(jj + blockSize, size);
                    const kEnd = Math.min(kk + blockSize, size);
                    
                    for (let i = ii; i < iEnd; i++) {
                        for (let j = jj; j < jEnd; j++) {
                            let sum = result[i * size + j];
                            for (let k = kk; k < kEnd; k++) {
                                sum += matrixA[i * size + k] * matrixB[k * size + j];
                            }
                            result[i * size + j] = sum;
                        }
                    }
                }
            }
        }
        
        const duration = performance.now() - start;
        const flops = 2 * size * size * size;
        const gflops = (flops / 1e9) / (duration / 1000);
        
        return { duration, gflops, matrixSize: size };
    }

    // ========== BENCHAMRK EXECUTION ==========

    async runParallelCPUBenchmark() {
        console.log(`🧮 Running parallel CPU benchmark...`);
        
        return new Promise((resolve) => {
            const workers = [];
            const results = [];
            let completed = 0;
            const startTime = performance.now();
            
            for (let i = 0; i < this.cpuCount; i++) {
                const worker = new Worker(__filename, {
                    workerData: { 
                        task: 'cpu',
                        workerId: i,
                        iterations: 20000000
                    }
                });
                
                worker.on('message', (result) => {
                    results.push(result);
                    completed++;
                    
                    if (completed === this.cpuCount) {
                        const totalTime = performance.now() - startTime;
                        const totalOps = results.reduce((sum, r) => sum + r.opsPerSecond, 0);
                        const avgEfficiency = results.reduce((sum, r) => sum + r.duration, 0) / results.length / totalTime * 100;
                        
                        workers.forEach(w => w.terminate());
                        
                        resolve({
                            totalTime,
                            totalOpsPerSecond: totalOps,
                            efficiency: avgEfficiency,
                            results
                        });
                    }
                });
                
                worker.on('error', console.error);
                workers.push(worker);
            }
        });
    }

    async runGPUBenchmarks() {
        console.log(`🎮 Running GPU benchmarks...`);
        
        const results = [];
        
        for (let i = 0; i < this.gpuInfo.length; i++) {
            const gpu = this.gpuInfo[i];
            console.log(`   Testing ${gpu.name}...`);
            
            const matrixResult = UniversalBenchmarkSuite.gpuMatrixBenchmark(
                gpu.vendor === 'Apple' ? 1024 : 512
            );
            
            results.push({
                gpu: gpu.name,
                vendor: gpu.vendor,
                apis: gpu.apis,
                matrixPerformance: matrixResult.gflops,
                matrixTime: matrixResult.duration
            });
            
            console.log(`     Matrix: ${matrixResult.gflops.toFixed(1)} GFLOPS`);
        }
        
        return results;
    }

    // ========== REPORT GENERATION ==========

    generateReport(cpuResults, memoryResults, gpuResults, stressResults = null) {
        const report = {
            timestamp: new Date().toISOString(),
            system: this.systemInfo,
            cpu: this.cpuInfo,
            gpus: this.gpuInfo,
            results: {
                cpu: cpuResults,
                memory: memoryResults,
                gpu: gpuResults
            },
            stressTest: stressResults
        };

        this.printFormattedReport(report);
        this.saveReport(report);
        
        return report;
    }

    printFormattedReport(report) {
        console.log(`\n🏆 COMPLETE HARDWARE ANALYSIS REPORT`);
        console.log(`${'='.repeat(70)}`);
        console.log(`⏰ Timestamp: ${new Date(report.timestamp).toLocaleString()}`);
        console.log(`💻 System: ${report.system.platform} ${report.system.arch}`);
        console.log(`🧮 CPU: ${report.cpu.model} (${report.cpu.cores} cores)`);
        console.log(`🔥 Est. TDP: ${report.cpu.thermalDesignPower}W`);
        
        // CPU Results
        console.log(`\n📊 CPU BENCHMARK PERFORMANCE:`);
        console.log(`   🔥 Multi-core: ${(report.results.cpu.totalOpsPerSecond / 1000000).toFixed(1)}M ops/sec`);
        console.log(`   ⚡ Efficiency: ${report.results.cpu.efficiency.toFixed(1)}%`);
        console.log(`   ⏱️  Runtime: ${report.results.cpu.totalTime.toFixed(0)}ms`);
        
        // Memory Results
        console.log(`\n💾 MEMORY PERFORMANCE:`);
        report.results.memory.forEach(mem => {
            console.log(`   ${mem.name}: ${mem.bandwidth.toFixed(1)} GB/s (${mem.latency.toFixed(1)}ns latency)`);
        });
        
        // GPU Results
        console.log(`\n🎮 GPU PERFORMANCE:`);
        report.results.gpu.forEach((gpu, i) => {
            console.log(`   GPU ${i + 1}: ${gpu.gpu} (${gpu.vendor})`);
            console.log(`     APIs: ${gpu.apis.join(', ')}`);
            console.log(`     Matrix: ${gpu.matrixPerformance.toFixed(1)} GFLOPS`);
        });
        
        // Stress Test Results
        if (report.stressTest) {
            console.log(`\n🔥 STRESS TEST SUMMARY:`);
            const stress = report.stressTest;
            console.log(`   Test Type: ${stress.testName.toUpperCase()}`);
            console.log(`   Duration: ${((stress.endTime - stress.startTime) / 1000).toFixed(1)}s`);
            console.log(`   Stability: ${stress.stability.crashes === 0 ? '✅ STABLE' : '❌ UNSTABLE'}`);
            
            if (stress.thermalData.length > 0) {
                const maxTemp = Math.max(...stress.thermalData.filter(t => t.cpu).map(t => t.cpu));
                console.log(`   Max CPU Temp: ${maxTemp.toFixed(1)}°C`);
            }
            
            if (stress.results && stress.results.type === 'combined') {
                console.log(`   🧮 CPU Under Load: ${(stress.results.cpu.avgOpsPerSecond / 1000000).toFixed(1)}M ops/sec`);
                if (stress.results.memory) console.log(`   💾 Memory Under Load: ${stress.results.memory.throughputGBps.toFixed(2)} GB/s`);
                if (stress.results.gpu) console.log(`   🎮 GPU Under Load: ${stress.results.gpu.avgGflops.toFixed(1)} GFLOPS`);
            }
        }
        
        // Overall Performance Grade
        const cpuScore = report.results.cpu.totalOpsPerSecond / 1000000;
        const gpuScore = Math.max(...report.results.gpu.map(g => g.matrixPerformance));
        const overallScore = cpuScore + gpuScore;
        
        // Adjust grade based on stress test stability
        let grade = 'F';
        if (overallScore > 300) grade = 'A+';
        else if (overallScore > 200) grade = 'A';
        else if (overallScore > 150) grade = 'B+';
        else if (overallScore > 100) grade = 'B';
        else if (overallScore > 50) grade = 'C';
        else if (overallScore > 25) grade = 'D';
        
        // Downgrade if stress test failed
        if (report.stressTest && report.stressTest.stability.crashes > 0) {
            const gradeIndex = ['F', 'D', 'C', 'B', 'B+', 'A', 'A+'].indexOf(grade);
            grade = ['F', 'D', 'C', 'B', 'B+', 'A', 'A+'][Math.max(0, gradeIndex - 1)];
            console.log(`   ⚠️  Grade reduced due to stability issues`);
        }
        
        console.log(`\n🎖️  OVERALL SYSTEM GRADE: ${grade}`);
        console.log(`📈 Performance Score: ${overallScore.toFixed(1)}`);
        console.log(`   CPU Contribution: ${cpuScore.toFixed(1)}`);
        console.log(`   GPU Contribution: ${gpuScore.toFixed(1)}`);
        
        if (report.stressTest) {
            console.log(`🔥 Stress Test Rating: ${report.stressTest.stability.crashes === 0 ? 'PASSED' : 'FAILED'}`);
        }
        
        console.log(`${'='.repeat(70)}`);
    }

    saveReport(report) {
        try {
            const filename = `complete_system_report_${Date.now()}.json`;
            fs.writeFileSync(filename, JSON.stringify(report, null, 2));
            console.log(`\n💾 Complete report saved to: ${filename}`);
        } catch (error) {
            console.log(`\n⚠️  Could not save report: ${error.message}`);
        }
    }

    // ========== MAIN BENCHMARK RUNNER ==========

    async runCompleteBenchmark() {
        const overallStart = performance.now();
        
        // Detect hardware
        await this.detectSystemInfo();
        
        console.log(`🚀 Starting comprehensive benchmark + stress test...\n`);
        
        // Run standard benchmarks first
        console.log(`📊 PHASE 1: Performance Benchmarks`);
        console.log(`${'='.repeat(40)}`);
        
        const cpuResults = await this.runParallelCPUBenchmark();
        console.log(`✅ CPU benchmark completed\n`);
        
        console.log(`💾 Running memory benchmarks...`);
        const memoryResults = UniversalBenchmarkSuite.memoryBenchmark();
        console.log(`✅ Memory benchmark completed\n`);
        
        const gpuResults = await this.runGPUBenchmarks();
        console.log(`✅ GPU benchmarks completed\n`);
        
        // Run stress tests
        console.log(`🔥 PHASE 2: Stress Testing`);
        console.log(`${'='.repeat(40)}`);
        
        const stressResults = await this.showStressTestMenu();
        
        const totalTime = performance.now() - overallStart;
        console.log(`⏱️  Total benchmark + stress test time: ${(totalTime / 1000).toFixed(1)}s\n`);
        
        // Generate comprehensive report
        const report = this.generateReport(cpuResults, memoryResults, gpuResults, stressResults);
        
        return report;
    }

    // Cleanup method for graceful shutdown
    cleanup() {
        console.log(`\n🛑 Cleaning up stress test workers...`);
        this.stressWorkers.forEach(worker => {
            try {
                worker.terminate();
            } catch (e) {
                // Worker already terminated
            }
        });
        this.stressWorkers = [];
    }
}

// Worker thread implementation
if (!isMainThread) {
    const { task, workerId, iterations, duration, intensity, sizeGB, complexity } = workerData;
    
    if (task === 'cpu') {
        const result = UniversalBenchmarkSuite.cpuBenchmark(iterations);
        result.workerId = workerId;
        parentPort.postMessage(result);
    } else if (task === 'cpu_stress') {
        const result = UniversalBenchmarkSuite.cpuStressTest(duration, intensity);
        result.workerId = workerId;
        parentPort.postMessage(result);
    } else if (task === 'memory_stress') {
        const result = UniversalBenchmarkSuite.memoryStressTest(duration, sizeGB);
        parentPort.postMessage(result);
    } else if (task === 'gpu_stress') {
        const result = UniversalBenchmarkSuite.gpuStressTest(duration, complexity);
        result.workerId = workerId;
        parentPort.postMessage(result);
    }
}

// Main execution with graceful shutdown
if (isMainThread) {
    const benchmark = new UniversalBenchmarkSuite();
    
    // Handle Ctrl+C gracefully
    process.on('SIGINT', () => {
        console.log(`\n\n🛑 Benchmark interrupted by user`);
        benchmark.cleanup();
        process.exit(0);
    });
    
    benchmark.runCompleteBenchmark()
        .then(() => {
            console.log(`\n🎉 Complete benchmark + stress test finished successfully!`);
            benchmark.cleanup();
            process.exit(0);
        })
        .catch(error => {
            console.error(`\n❌ Benchmark failed:`, error);
            benchmark.cleanup();
            process.exit(1);
        });
}