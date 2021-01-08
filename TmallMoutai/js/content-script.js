// 确认插件加载
console.log('Tmall Moutai Activate')

// 提交抢购时间
buyTime = '20:20:00'

// 记录当前页面
var current = 0

// 记录间隔触发器ID
var intervalId = 0

// 延迟1秒开始执行程序
setTimeout(function(){
    // 判断页面当面页面是购物车还是订单提交或是滑块验证界面
    var cart = document.getElementById('J_Go')
    var order = document.getElementsByClassName('go-btn')[0]
	var verify = document.getElementById("nc_1_n1z")

    if (cart != null || cart != undefined) {
        console.log('目前在购物车页面')
        current =  1
    }

    if (order != null || order != undefined) {
        console.log('目前在订单提交页面')
        current =  2
    }
	
	if (verify != null || verify != undefined) {
        console.log('目前在滑块验证页面')
        current =  3
    }

    // 开始时间监听
    globalIntervalId = setInterval("timeListener()","10")
}, 500)


// 时间监听
function timeListener() {
	if (current == 3) {
		// 当前在滑块验证页面
		// 停止时间监听
        clearInterval(globalIntervalId)
        // 开始滑块验证
        mockVerify()
	} else if (current == 2) {
		// 当前在订单提交页面
        // 停止时间监听
        clearInterval(globalIntervalId)
        // 提交订单
        orderGo()
    } else if (current == 1) {
        // 当前在购物车页面
		// 获取当前时间
        var date = new Date()
        var timeStr = date.toLocaleString('chinese',{hour12:false}).split(' ')[1]
        console.log(timeStr)
        // 到达预定时间
        if (timeStr == buyTime) {
			// 停止时间监听
			clearInterval(globalIntervalId)
            console.log('时间已到，开始抢购')
            cartGo()
        }
    } else {
        // 不在淘宝
        console.log('不在淘宝或页面尚未加载')
        // 停止时间监听
        // clearInterval(globalIntervalId)
    }
}

// 结算
function cartGo() {
    document.getElementById('J_Go').click()
}

// 提交订单
function orderGo() {
    document.getElementsByClassName('go-btn')[0].click()
}

// 滑块验证
function mockVerify() {
	// 获取滑块元素
	var btn=document.getElementById("nc_1_n1z")
	// 创建鼠标按下事件
	var mousedown = document.createEvent("MouseEvents")
	// 获取滑块位置
	var rect = btn.getBoundingClientRect()
	// 滑块x轴位置
	var x = rect.x
	// 滑块y轴位置
	var y = rect.y
	// 初始化鼠标按下事件
	mousedown.initMouseEvent("mousedown", true, true, window, 0, x, y, x, y, false, false, false, false, 0, null)
	// 触发鼠标按下事件
	btn.dispatchEvent(mousedown)

	// 记录鼠标向右移动的累计距离
	sum_x = 0
	// 每30ms执行一次鼠标移动，模拟鼠标拖动效果
	var  verifyIntervalId = setInterval(function(){
		// 创建鼠标移动事件
		var mousemove = document.createEvent("MouseEvents")
		// 计算鼠标目前所在位置
		var now_x = x + sum_x
		// 初始化鼠标移动事件
		mousemove.initMouseEvent("mousemove", true, true, window, 0, now_x, y, now_x, y, false, false, false, false, 0, null)
		// 触发鼠标移动事件
		btn.dispatchEvent(mousemove)
		// 触发鼠标移动事件
		// btn.dispatchEvent(mousemove)
		// 判断滑块是否已经达到最右端
		if(now_x - x >= 300){  // 已经到达最右端
			// 停止重复鼠标移动
			clearInterval(verifyIntervalId)
			// 创建鼠标抬起事件
			var mouseup = document.createEvent("MouseEvents")
			// 初始化鼠标抬起事件
			mouseup.initMouseEvent("mouseup", true, true, window, 0, now_x, y, now_x, y, false, false, false, false, 0, null)
			// 触发鼠标抬起事件
			btn.dispatchEvent(mouseup);
		}
		else{  // 还未达到最右端
			// 向右移动50px
			sum_x += 50
		}
	}, 30)
}















