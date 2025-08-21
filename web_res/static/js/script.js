// 自学习插件管理后台 - ECharts可视化大屏

// 登录状态检查
async function checkAuthStatus() {
    try {
        const response = await fetch('/api/config');
        if (response.status === 401) {
            window.location.href = '/api/login';
            return false;
        }
        return true;
    } catch (error) {
        console.error('检查认证状态失败:', error);
        return false;
    }
}

// 登出功能
async function logout() {
    try {
        const response = await fetch('/api/logout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            window.location.href = '/api/login';
        } else {
            console.error('登出失败');
        }
    } catch (error) {
        console.error('登出请求失败:', error);
    }
}

// 全局变量
let currentConfig = {};
let currentMetrics = {};
let chartInstances = {};

// ECharts Google Material Design 主题
const materialTheme = {
    color: ['#1976d2', '#4caf50', '#ff9800', '#f44336', '#9c27b0', '#00bcd4', '#795548', '#607d8b'],
    backgroundColor: 'transparent',
    textStyle: {
        fontFamily: 'Roboto, sans-serif',
        fontSize: 12,
        color: '#424242'
    },
    title: {
        textStyle: {
            fontFamily: 'Roboto, sans-serif',
            fontSize: 16,
            fontWeight: 500,
            color: '#212121'
        }
    },
    legend: {
        textStyle: {
            fontFamily: 'Roboto, sans-serif',
            fontSize: 12,
            color: '#757575'
        }
    },
    categoryAxis: {
        axisLine: { lineStyle: { color: '#e0e0e0' } },
        axisTick: { lineStyle: { color: '#e0e0e0' } },
        axisLabel: { color: '#757575' },
        splitLine: { lineStyle: { color: '#f5f5f5' } }
    },
    valueAxis: {
        axisLine: { lineStyle: { color: '#e0e0e0' } },
        axisTick: { lineStyle: { color: '#e0e0e0' } },
        axisLabel: { color: '#757575' },
        splitLine: { lineStyle: { color: '#f5f5f5' } }
    },
    grid: {
        borderColor: '#e0e0e0'
    }
};

// 初始化应用
document.addEventListener('DOMContentLoaded', async () => {
    console.log('自学习插件管理后台加载中...');
    
    // 首先检查认证状态
    const isAuthenticated = await checkAuthStatus();
    if (!isAuthenticated) {
        return; // 如果未认证，停止加载
    }
    
    // 绑定登出按钮事件
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', logout);
    }
    
    // 注册ECharts主题
    echarts.registerTheme('material', materialTheme);
    
    // 初始化菜单导航
    initializeNavigation();
    
    // 加载初始数据
    await loadInitialData();
    
    // 初始化可视化大屏
    initializeDashboard();
    
    // 设置定时刷新
    setInterval(refreshDashboard, 30000); // 每30秒刷新一次
    
    console.log('管理后台初始化完成');
});

// 初始化导航菜单
function initializeNavigation() {
    const menuItems = document.querySelectorAll('.menu-item');
    const pages = document.querySelectorAll('.page');
    
    menuItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetPage = item.getAttribute('data-page');
            
            // 更新菜单状态
            menuItems.forEach(mi => mi.classList.remove('active'));
            item.classList.add('active');
            
            // 显示对应页面
            pages.forEach(page => page.classList.remove('active'));
            const targetPageElement = document.getElementById(targetPage + '-page');
            if (targetPageElement) {
                targetPageElement.classList.add('active');
            }
            
            // 更新页面标题和面包屑
            const pageTitle = item.querySelector('span').textContent;
            document.getElementById('page-title').textContent = pageTitle;
            document.getElementById('current-page').textContent = pageTitle;
            
            // 加载页面数据
            loadPageData(targetPage);
        });
    });
}

// 加载初始数据
async function loadInitialData() {
    updateRefreshIndicator('加载中...');
    try {
        await Promise.all([
            loadConfig(),
            loadMetrics(),
            loadPersonaUpdates(),
            loadLearningStatus()
        ]);
        
        updateRefreshIndicator('刚刚更新');
    } catch (error) {
        console.error('加载初始数据失败:', error);
        showError('加载数据失败，请刷新页面重试');
        updateRefreshIndicator('更新失败');
    }
}

// 初始化可视化大屏
function initializeDashboard() {
    // 渲染概览统计
    renderOverviewStats();
    
    // 初始化所有图表
    initializeCharts();
    
    // 绑定控件事件
    bindChartControls();
}

// 渲染概览统计
function renderOverviewStats() {
    const stats = currentMetrics;
    
    // 更新统计数字
    document.getElementById('total-messages').textContent = formatNumber(stats.total_messages_collected || 0);
    document.getElementById('filtered-messages').textContent = formatNumber(stats.filtered_messages || 0);
    
    // 计算总LLM调用次数
    const totalLLMCalls = Object.values(stats.llm_calls || {}).reduce((sum, model) => sum + (model.total_calls || 0), 0);
    document.getElementById('total-llm-calls').textContent = formatNumber(totalLLMCalls);
    
    // 模拟学习会话数
    document.getElementById('learning-sessions').textContent = formatNumber(Math.floor(totalLLMCalls / 10) || 0);
}

// 初始化图表
function initializeCharts() {
    // LLM使用分布饼图
    initializeLLMUsagePie();
    
    // 消息处理趋势线图
    initializeMessageTrendLine();
    
    // LLM响应时间柱状图
    initializeResponseTimeBar();
    
    // 学习进度仪表盘
    initializeLearningProgressGauge();
    
    // 系统状态雷达图
    initializeSystemStatusRadar();
    
    // 用户活跃度热力图
    initializeActivityHeatmap();
}

// LLM使用分布饼图
function initializeLLMUsagePie() {
    const chartDom = document.getElementById('llm-usage-pie');
    const chart = echarts.init(chartDom, 'material');
    chartInstances['llm-usage-pie'] = chart;
    
    const llmData = currentMetrics.llm_calls || {};
    const data = Object.entries(llmData).map(([model, stats]) => ({
        name: model,
        value: stats.total_calls || 0
    }));
    
    const option = {
        tooltip: {
            trigger: 'item',
            formatter: '{a} <br/>{b}: {c} ({d}%)'
        },
        legend: {
            bottom: '5%',
            left: 'center'
        },
        series: [
            {
                name: 'LLM调用分布',
                type: 'pie',
                radius: ['40%', '70%'],
                center: ['50%', '45%'],
                data: data,
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowOffsetX: 0,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                },
                label: {
                    show: true,
                    formatter: '{b}: {c}'
                },
                labelLine: {
                    show: true
                }
            }
        ]
    };
    
    chart.setOption(option);
}

// 消息处理趋势线图
function initializeMessageTrendLine() {
    const chartDom = document.getElementById('message-trend-line');
    const chart = echarts.init(chartDom, 'material');
    chartInstances['message-trend-line'] = chart;
    
    // 使用真实数据或从API获取趋势数据
    fetch('/api/analytics/trends')
        .then(response => response.json())
        .then(data => {
            const hourlyData = data.hourly_trends || [];
            const hours = hourlyData.map(item => item.time);
            const rawMessages = hourlyData.map(item => item.raw_messages);
            const filteredMessages = hourlyData.map(item => item.filtered_messages);
            
            const option = {
                tooltip: {
                    trigger: 'axis',
                    axisPointer: {
                        type: 'cross'
                    }
                },
                legend: {
                    data: ['原始消息', '筛选消息']
                },
                xAxis: {
                    type: 'category',
                    data: hours,
                    boundaryGap: false
                },
                yAxis: {
                    type: 'value'
                },
                series: [
                    {
                        name: '原始消息',
                        type: 'line',
                        data: rawMessages,
                        smooth: true,
                        itemStyle: { color: '#2196f3' },
                        areaStyle: { opacity: 0.3 }
                    },
                    {
                        name: '筛选消息',
                        type: 'line',
                        data: filteredMessages,
                        smooth: true,
                        itemStyle: { color: '#4caf50' },
                        areaStyle: { opacity: 0.3 }
                    }
                ]
            };
            
            chart.setOption(option);
        })
        .catch(error => {
            console.error('加载趋势数据失败:', error);
            // 使用模拟数据作为后备
            initializeMessageTrendLineWithMockData(chart);
        });
}

// 后备的模拟数据方法
function initializeMessageTrendLineWithMockData(chart) {
    const hours = [];
    const rawMessages = [];
    const filteredMessages = [];
    
    for (let i = 23; i >= 0; i--) {
        const hour = new Date(Date.now() - i * 60 * 60 * 1000);
        hours.push(hour.getHours() + ':00');
        rawMessages.push(Math.floor(Math.random() * 50) + 10);
        filteredMessages.push(Math.floor(Math.random() * 20) + 5);
    }
    
    const option = {
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'cross'
            }
        },
        legend: {
            data: ['原始消息', '筛选消息']
        },
        xAxis: {
            type: 'category',
            data: hours,
            boundaryGap: false
        },
        yAxis: {
            type: 'value'
        },
        series: [
            {
                name: '原始消息',
                type: 'line',
                data: rawMessages,
                smooth: true,
                itemStyle: { color: '#2196f3' }
            },
            {
                name: '筛选消息',
                type: 'line',
                data: filteredMessages,
                smooth: true,
                itemStyle: { color: '#4caf50' }
            }
        ]
    };
    
    chart.setOption(option);
}

// LLM响应时间柱状图
function initializeResponseTimeBar() {
    const chartDom = document.getElementById('response-time-bar');
    const chart = echarts.init(chartDom, 'material');
    chartInstances['response-time-bar'] = chart;
    
    const llmData = currentMetrics.llm_calls || {};
    const models = Object.keys(llmData);
    const responseTimes = Object.values(llmData).map(stats => stats.avg_response_time_ms || 0);
    
    const option = {
        tooltip: {
            trigger: 'axis',
            formatter: '{b}<br/>{a}: {c}ms'
        },
        xAxis: {
            type: 'category',
            data: models,
            axisLabel: {
                rotate: 45
            }
        },
        yAxis: {
            type: 'value',
            name: '响应时间(ms)'
        },
        series: [
            {
                name: '平均响应时间',
                type: 'bar',
                data: responseTimes,
                itemStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: '#1976d2' },
                        { offset: 1, color: '#64b5f6' }
                    ])
                },
                markLine: {
                    data: [
                        { type: 'average', name: '平均值' }
                    ]
                }
            }
        ]
    };
    
    chart.setOption(option);
}

// 学习进度仪表盘
function initializeLearningProgressGauge() {
    const chartDom = document.getElementById('learning-progress-gauge');
    const chart = echarts.init(chartDom, 'material');
    chartInstances['learning-progress-gauge'] = chart;
    
    // 计算学习效率
    const totalMessages = currentMetrics.total_messages_collected || 0;
    const filteredMessages = currentMetrics.filtered_messages || 0;
    const efficiency = totalMessages > 0 ? (filteredMessages / totalMessages * 100) : 0;
    
    const option = {
        series: [
            {
                type: 'gauge',
                startAngle: 180,
                endAngle: 0,
                center: ['50%', '75%'],
                radius: '90%',
                min: 0,
                max: 100,
                splitNumber: 8,
                axisLine: {
                    lineStyle: {
                        width: 6,
                        color: [
                            [0.25, '#ff4444'],
                            [0.5, '#ff9800'],
                            [0.75, '#4caf50'],
                            [1, '#1976d2']
                        ]
                    }
                },
                pointer: {
                    icon: 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
                    length: '12%',
                    width: 20,
                    offsetCenter: [0, '-60%'],
                    itemStyle: {
                        color: 'auto'
                    }
                },
                axisTick: {
                    length: 12,
                    lineStyle: {
                        color: 'auto',
                        width: 2
                    }
                },
                splitLine: {
                    length: 20,
                    lineStyle: {
                        color: 'auto',
                        width: 5
                    }
                },
                axisLabel: {
                    color: '#464646',
                    fontSize: 10,
                    distance: -60,
                    formatter: function (value) {
                        if (value === 100) {
                            return '优秀';
                        } else if (value === 75) {
                            return '良好';
                        } else if (value === 50) {
                            return '一般';
                        } else if (value === 25) {
                            return '较差';
                        }
                        return '';
                    }
                },
                title: {
                    offsetCenter: [0, '-10%'],
                    fontSize: 16
                },
                detail: {
                    fontSize: 30,
                    offsetCenter: [0, '-35%'],
                    valueAnimation: true,
                    formatter: function (value) {
                        return Math.round(value) + '%';
                    },
                    color: 'auto'
                },
                data: [
                    {
                        value: efficiency.toFixed(1),
                        name: '学习效率'
                    }
                ]
            }
        ]
    };
    
    chart.setOption(option);
}

// 系统状态雷达图
function initializeSystemStatusRadar() {
    const chartDom = document.getElementById('system-status-radar');
    const chart = echarts.init(chartDom, 'material');
    chartInstances['system-status-radar'] = chart;
    
    const option = {
        tooltip: {},
        radar: {
            indicator: [
                { name: '消息抓取', max: 100 },
                { name: '数据筛选', max: 100 },
                { name: 'LLM调用', max: 100 },
                { name: '学习质量', max: 100 },
                { name: '响应速度', max: 100 },
                { name: '系统稳定性', max: 100 }
            ],
            center: ['50%', '50%'],
            radius: '75%'
        },
        series: [
            {
                name: '系统状态',
                type: 'radar',
                data: [
                    {
                        value: [85, 92, 78, 88, 82, 95],
                        name: '当前状态',
                        itemStyle: { color: '#1976d2' },
                        areaStyle: { opacity: 0.3 }
                    }
                ]
            }
        ]
    };
    
    chart.setOption(option);
}

// 用户活跃度热力图
function initializeActivityHeatmap() {
    const chartDom = document.getElementById('activity-heatmap');
    const chart = echarts.init(chartDom, 'material');
    chartInstances['activity-heatmap'] = chart;
    
    // 从API获取热力图数据
    fetch('/api/analytics/trends')
        .then(response => response.json())
        .then(data => {
            const heatmapData = data.activity_heatmap || {};
            const actualData = heatmapData.data || [];
            const days = heatmapData.days || ['周一', '周二', '周三', '周四', '周五', '周六', '周日'];
            const hours = heatmapData.hours || [];
            
            const option = {
                tooltip: {
                    position: 'top',
                    formatter: function (params) {
                        return `${days[params.value[1]]} ${hours[params.value[0]]}<br/>活跃度: ${params.value[2]}`;
                    }
                },
                grid: {
                    height: '50%',
                    top: '10%'
                },
                xAxis: {
                    type: 'category',
                    data: hours,
                    splitArea: {
                        show: true
                    }
                },
                yAxis: {
                    type: 'category',
                    data: days,
                    splitArea: {
                        show: true
                    }
                },
                visualMap: {
                    min: 0,
                    max: 50,
                    calculable: true,
                    orient: 'horizontal',
                    left: 'center',
                    bottom: '15%',
                    inRange: {
                        color: ['#e3f2fd', '#1976d2']
                    }
                },
                series: [
                    {
                        name: '活跃度',
                        type: 'heatmap',
                        data: actualData,
                        label: {
                            show: false
                        },
                        emphasis: {
                            itemStyle: {
                                shadowBlur: 10,
                                shadowColor: 'rgba(0, 0, 0, 0.5)'
                            }
                        }
                    }
                ]
            };
            
            chart.setOption(option);
        })
        .catch(error => {
            console.error('加载活跃度数据失败:', error);
            // 使用模拟数据作为后备
            initializeActivityHeatmapWithMockData(chart);
        });
}

// 后备的活跃度热力图
function initializeActivityHeatmapWithMockData(chart) {
    const hours = [];
    const days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'];
    for (let i = 0; i < 24; i++) {
        hours.push(i + ':00');
    }
    
    const data = [];
    for (let i = 0; i < 7; i++) {
        for (let j = 0; j < 24; j++) {
            data.push([j, i, Math.floor(Math.random() * 50)]);
        }
    }
    
    const option = {
        tooltip: {
            position: 'top',
            formatter: function (params) {
                return `${days[params.value[1]]} ${hours[params.value[0]]}<br/>活跃度: ${params.value[2]}`;
            }
        },
        grid: {
            height: '50%',
            top: '10%'
        },
        xAxis: {
            type: 'category',
            data: hours,
            splitArea: {
                show: true
            }
        },
        yAxis: {
            type: 'category',
            data: days,
            splitArea: {
                show: true
            }
        },
        visualMap: {
            min: 0,
            max: 50,
            calculable: true,
            orient: 'horizontal',
            left: 'center',
            bottom: '15%',
            inRange: {
                color: ['#e3f2fd', '#1976d2']
            }
        },
        series: [
            {
                name: '活跃度',
                type: 'heatmap',
                data: data,
                label: {
                    show: false
                },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }
        ]
    };
    
    chart.setOption(option);
}

// 绑定图表控件事件
function bindChartControls() {
    // LLM时间范围选择器
    document.getElementById('llm-time-range').addEventListener('change', (e) => {
        updateLLMUsageChart(e.target.value);
    });
    
    // 消息时间范围选择器
    document.getElementById('message-time-range').addEventListener('change', (e) => {
        updateMessageTrendChart(e.target.value);
    });
    
    // 活跃度时间按钮
    document.querySelectorAll('.time-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            // 更新按钮状态
            document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            
            // 更新热力图
            updateActivityHeatmap(e.target.dataset.period);
        });
    });
    
    // 配置保存按钮
    const saveBtn = document.getElementById('saveConfig');
    if (saveBtn) {
        saveBtn.addEventListener('click', saveConfiguration);
    }
    
    // 配置重置按钮
    const resetBtn = document.getElementById('resetConfig');
    if (resetBtn) {
        resetBtn.addEventListener('click', resetConfiguration);
    }
}

// 加载配置数据
async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        if (response.ok) {
            currentConfig = await response.json();
            renderConfigPage();
        } else {
            throw new Error('加载配置失败');
        }
    } catch (error) {
        console.error('加载配置失败:', error);
    }
}

// 加载性能指标
async function loadMetrics() {
    try {
        const response = await fetch('/api/metrics');
        if (response.ok) {
            currentMetrics = await response.json();
        } else {
            throw new Error('加载性能指标失败');
        }
    } catch (error) {
        console.error('加载性能指标失败:', error);
    }
}

// 加载人格更新数据
async function loadPersonaUpdates() {
    try {
        const response = await fetch('/api/persona_updates');
        if (response.ok) {
            const updates = await response.json();
            renderPersonaUpdates(updates);
            updateReviewStats(updates);
        } else {
            throw new Error('加载人格更新失败');
        }
    } catch (error) {
        console.error('加载人格更新失败:', error);
    }
}

// 加载学习状态
async function loadLearningStatus() {
    try {
        // 模拟学习状态数据
        const mockStatus = {
            current_session: {
                session_id: 'sess_' + Date.now(),
                start_time: new Date(Date.now() - 2 * 60 * 60 * 1000).toLocaleString(),
                messages_processed: Math.floor(Math.random() * 100) + 50,
                status: Math.random() > 0.5 ? 'active' : 'stopped'
            }
        };
        
        renderLearningStatus(mockStatus);
    } catch (error) {
        console.error('加载学习状态失败:', error);
    }
}

// 渲染配置页面
function renderConfigPage() {
    // 更新开关状态
    document.getElementById('enableMessageCapture').checked = currentConfig.enable_message_capture || false;
    document.getElementById('enableAutoLearning').checked = currentConfig.enable_auto_learning || false;
    document.getElementById('enableRealtimeLearning').checked = currentConfig.enable_realtime_learning || false;
    
    // 更新其他配置项
    if (currentConfig.target_qq_list) {
        document.getElementById('targetQQList').value = currentConfig.target_qq_list.join(', ');
    }
    
    if (currentConfig.learning_interval_hours) {
        document.getElementById('learningInterval').value = currentConfig.learning_interval_hours;
    }
    
    if (currentConfig.filter_model_name) {
        document.getElementById('filterModel').value = currentConfig.filter_model_name;
    }
    
    if (currentConfig.refine_model_name) {
        document.getElementById('refineModel').value = currentConfig.refine_model_name;
    }
}

// 渲染人格更新列表
function renderPersonaUpdates(updates) {
    const reviewList = document.getElementById('review-list');
    
    if (!updates || updates.length === 0) {
        reviewList.innerHTML = '<div class="no-updates">暂无待审查的人格更新</div>';
        return;
    }
    
    reviewList.innerHTML = updates.map(update => `
        <div class="persona-update-item">
            <div class="update-content">
                <h4>更新 ID: ${update.id}</h4>
                <p><strong>原因:</strong> ${update.reason || '未提供'}</p>
                <p><strong>时间:</strong> ${new Date(update.timestamp * 1000).toLocaleString()}</p>
                <p><strong>内容:</strong> ${update.content || '未提供'}</p>
            </div>
            <div class="update-actions">
                <button class="btn btn-success" onclick="reviewUpdate(${update.id}, 'approve')">
                    <i class="material-icons">check</i>
                    批准
                </button>
                <button class="btn btn-danger" onclick="reviewUpdate(${update.id}, 'reject')">
                    <i class="material-icons">close</i>
                    拒绝
                </button>
            </div>
        </div>
    `).join('');
}

// 更新审查统计
function updateReviewStats(updates) {
    const pending = updates.filter(u => !u.reviewed).length;
    const approved = updates.filter(u => u.reviewed && u.approved).length;
    const rejected = updates.filter(u => u.reviewed && !u.approved).length;
    
    document.getElementById('pending-reviews').textContent = pending;
    document.getElementById('approved-reviews').textContent = approved;
    document.getElementById('rejected-reviews').textContent = rejected;
}

// 渲染学习状态
function renderLearningStatus(status) {
    const session = status.current_session;
    if (session) {
        document.getElementById('current-session-id').textContent = session.session_id;
        document.getElementById('session-start-time').textContent = session.start_time;
        document.getElementById('session-messages').textContent = session.messages_processed;
        
        const statusBadge = document.getElementById('session-status');
        statusBadge.textContent = session.status === 'active' ? '运行中' : '已停止';
        statusBadge.className = `status-badge ${session.status === 'active' ? 'active' : ''}`;
    }
}

// 保存配置
async function saveConfiguration() {
    const newConfig = {
        enable_message_capture: document.getElementById('enableMessageCapture').checked,
        enable_auto_learning: document.getElementById('enableAutoLearning').checked,
        enable_realtime_learning: document.getElementById('enableRealtimeLearning').checked,
        target_qq_list: document.getElementById('targetQQList').value.split(',').map(qq => qq.trim()).filter(qq => qq),
        learning_interval_hours: parseInt(document.getElementById('learningInterval').value),
        filter_model_name: document.getElementById('filterModel').value,
        refine_model_name: document.getElementById('refineModel').value
    };
    
    try {
        showSpinner(document.getElementById('saveConfig'));
        
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(newConfig)
        });
        
        if (response.ok) {
            const result = await response.json();
            currentConfig = result.new_config;
            showSuccess('配置保存成功');
            
            // 更新仪表盘
            setTimeout(() => {
                renderOverviewStats();
                updateSystemStatusRadar();
            }, 1000);
        } else {
            throw new Error('保存配置失败');
        }
    } catch (error) {
        console.error('保存配置失败:', error);
        showError('保存配置失败，请重试');
    } finally {
        hideSpinner(document.getElementById('saveConfig'));
    }
}

// 重置配置
async function resetConfiguration() {
    if (confirm('确定要重置配置到默认值吗？')) {
        try {
            // 重置表单到默认值
            document.getElementById('enableMessageCapture').checked = true;
            document.getElementById('enableAutoLearning').checked = true;
            document.getElementById('enableRealtimeLearning').checked = false;
            document.getElementById('targetQQList').value = '';
            document.getElementById('learningInterval').value = 6;
            document.getElementById('filterModel').value = 'gpt-4o-mini';
            document.getElementById('refineModel').value = 'gpt-4o';
            
            showSuccess('配置已重置到默认值');
        } catch (error) {
            showError('重置配置失败');
        }
    }
}

// 审查人格更新
async function reviewUpdate(updateId, action) {
    try {
        const response = await fetch(`/api/persona_updates/${updateId}/review`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ action })
        });
        
        if (response.ok) {
            showSuccess(`人格更新已${action === 'approve' ? '批准' : '拒绝'}`);
            await loadPersonaUpdates(); // 重新加载列表
        } else {
            throw new Error('审查操作失败');
        }
    } catch (error) {
        console.error('审查操作失败:', error);
        showError('操作失败，请重试');
    }
}

// 更新LLM使用图表
function updateLLMUsageChart(timeRange) {
    // 模拟根据时间范围更新数据
    console.log('更新LLM使用图表:', timeRange);
    if (chartInstances['llm-usage-pie']) {
        // 这里可以重新获取数据并更新图表
        initializeLLMUsagePie();
    }
}

// 更新消息趋势图表
function updateMessageTrendChart(timeRange) {
    console.log('更新消息趋势图表:', timeRange);
    if (chartInstances['message-trend-line']) {
        initializeMessageTrendLine();
    }
}

// 更新活跃度热力图
function updateActivityHeatmap(period) {
    console.log('更新活跃度热力图:', period);
    if (chartInstances['activity-heatmap']) {
        initializeActivityHeatmap();
    }
}

// 更新系统状态雷达图
function updateSystemStatusRadar() {
    if (chartInstances['system-status-radar']) {
        // 根据当前配置更新状态值
        const values = [
            currentConfig.enable_message_capture ? 95 : 0,
            85, 78, 88, 82, 95
        ];
        
        const option = chartInstances['system-status-radar'].getOption();
        option.series[0].data[0].value = values;
        chartInstances['system-status-radar'].setOption(option);
    }
}

// 加载页面数据
async function loadPageData(page) {
    switch (page) {
        case 'dashboard':
            await loadMetrics();
            renderOverviewStats();
            // 更新所有图表
            Object.values(chartInstances).forEach(chart => {
                if (chart && typeof chart.resize === 'function') {
                    setTimeout(() => chart.resize(), 100);
                }
            });
            break;
        case 'config':
            await loadConfig();
            break;
        case 'persona-review':
            await loadPersonaUpdates();
            break;
        case 'learning-status':
            await loadLearningStatus();
            break;
        case 'metrics':
            await loadMetrics();
            renderDetailedMetrics();
            break;
    }
}

// 渲染详细监控
function renderDetailedMetrics() {
    // API监控图表
    initializeAPIMetricsChart();
    
    // 数据库监控图表
    initializeDBMetricsChart();
    
    // 内存使用图表
    initializeMemoryMetricsChart();
}

// API监控图表
function initializeAPIMetricsChart() {
    const chartDom = document.getElementById('api-metrics-chart');
    if (!chartDom) return;
    
    const chart = echarts.init(chartDom, 'material');
    chartInstances['api-metrics-chart'] = chart;
    
    const option = {
        tooltip: {
            trigger: 'axis'
        },
        xAxis: {
            type: 'category',
            data: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00']
        },
        yAxis: {
            type: 'value',
            name: '响应时间(ms)'
        },
        series: [
            {
                name: 'API响应时间',
                type: 'line',
                data: [120, 132, 101, 134, 90, 230],
                smooth: true,
                itemStyle: { color: '#1976d2' }
            }
        ]
    };
    
    chart.setOption(option);
}

// 数据库监控图表
function initializeDBMetricsChart() {
    const chartDom = document.getElementById('db-metrics-chart');
    if (!chartDom) return;
    
    const chart = echarts.init(chartDom, 'material');
    chartInstances['db-metrics-chart'] = chart;
    
    const option = {
        tooltip: {
            trigger: 'axis'
        },
        xAxis: {
            type: 'category',
            data: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00']
        },
        yAxis: {
            type: 'value',
            name: '查询时间(ms)'
        },
        series: [
            {
                name: '数据库查询',
                type: 'bar',
                data: [20, 25, 18, 30, 22, 28],
                itemStyle: { color: '#4caf50' }
            }
        ]
    };
    
    chart.setOption(option);
}

// 内存使用图表
function initializeMemoryMetricsChart() {
    const chartDom = document.getElementById('memory-metrics-chart');
    if (!chartDom) return;
    
    const chart = echarts.init(chartDom, 'material');
    chartInstances['memory-metrics-chart'] = chart;
    
    const option = {
        tooltip: {
            trigger: 'axis',
            formatter: '{b}<br/>{a}: {c}%'
        },
        xAxis: {
            type: 'category',
            data: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00']
        },
        yAxis: {
            type: 'value',
            name: '使用率(%)',
            max: 100
        },
        series: [
            {
                name: '内存使用率',
                type: 'line',
                data: [45, 52, 48, 60, 55, 58],
                smooth: true,
                areaStyle: { opacity: 0.3 },
                itemStyle: { color: '#ff9800' }
            }
        ]
    };
    
    chart.setOption(option);
}

// 学习历史图表
function initializeLearningHistoryChart() {
    const chartDom = document.getElementById('learning-history-chart');
    if (!chartDom) return;
    
    const chart = echarts.init(chartDom, 'material');
    chartInstances['learning-history-chart'] = chart;
    
    const option = {
        tooltip: {
            trigger: 'axis'
        },
        xAxis: {
            type: 'category',
            data: ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        },
        yAxis: {
            type: 'value',
            name: '学习次数'
        },
        series: [
            {
                name: '学习会话',
                type: 'bar',
                data: [12, 15, 8, 20, 18, 6, 9],
                itemStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: '#9c27b0' },
                        { offset: 1, color: '#e1bee7' }
                    ])
                }
            }
        ]
    };
    
    chart.setOption(option);
}

// 刷新仪表盘
async function refreshDashboard() {
    if (document.querySelector('#dashboard-page.active')) {
        updateRefreshIndicator('更新中...', true);
        
        try {
            await loadMetrics();
            renderOverviewStats();
            
            // 更新图表数据
            initializeLLMUsagePie();
            initializeMessageTrendLine();
            initializeResponseTimeBar();
            initializeLearningProgressGauge();
            
            updateRefreshIndicator('刚刚更新');
        } catch (error) {
            console.error('刷新失败:', error);
            updateRefreshIndicator('更新失败');
        }
    }
}

// 更新刷新指示器
function updateRefreshIndicator(text, spinning = false) {
    const indicator = document.getElementById('last-update');
    const icon = document.querySelector('.refresh-icon');
    
    if (indicator) indicator.textContent = text;
    if (icon) {
        if (spinning) {
            icon.classList.add('spinning');
        } else {
            icon.classList.remove('spinning');
        }
    }
}

// 工具函数
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

function showSpinner(element) {
    const originalText = element.innerHTML;
    element.innerHTML = '<div class="loading"></div> 保存中...';
    element.disabled = true;
    element.dataset.originalText = originalText;
}

function hideSpinner(element) {
    element.innerHTML = element.dataset.originalText || element.innerHTML;
    element.disabled = false;
}

function showSuccess(message) {
    showNotification(message, 'success');
}

function showError(message) {
    showNotification(message, 'error');
}

function showNotification(message, type) {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        document.body.removeChild(notification);
    }, 3000);
}

// 窗口大小改变时重新调整图表大小
window.addEventListener('resize', () => {
    Object.values(chartInstances).forEach(chart => {
        if (chart && typeof chart.resize === 'function') {
            chart.resize();
        }
    });
});

// 页面可见性改变时暂停/恢复刷新
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // 页面隐藏时可以暂停定时器
        console.log('页面隐藏，暂停刷新');
    } else {
        // 页面显示时可以立即刷新一次
        console.log('页面显示，恢复刷新');
        refreshDashboard();
    }
});