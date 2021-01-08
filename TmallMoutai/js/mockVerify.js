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
var  interval = setInterval(function(){
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
        clearInterval(interval)
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
}, 30);