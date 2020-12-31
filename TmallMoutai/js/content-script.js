// 确认插件加载
console.log('Tmall Moutai Activate')

// 提交抢购时间
buyTime = '20:00:00'

// 记录当前页面
var current = 0

// 记录间隔触发器ID
var intervalId = 0

// 延迟1秒开始执行程序
setTimeout(function(){
    // 判断页面当面页面是购物车还是订单提交
    var cart = document.getElementById('J_Go')
    var order = document.getElementsByClassName('go-btn')[0]

    if (cart != null || cart != undefined) {
        console.log('目前在购物车页面')
        current =  1
    }

    if (order != null || order != undefined) {
        console.log('目前在订单提交页面')
        current =  2
    }

    // 开始时间监听
    intervalId = setInterval("timeListener()","100");

}, 1000);


// 时间监听
function timeListener() {
    // 当前在订单提交页面
    if (current == 2) {
        // 停止时间监听
        clearInterval(intervalId)
        // 提交订单
        orderGo()
    } else if (current == 1) {
        // 当前在购物车页面
        var date = new Date()
        var timeStr = date.toLocaleString('chinese',{hour12:false}).split(' ')[1]
        console.log(timeStr)
        // 到达预定时间
        if (timeStr == buyTime) {
			// 停止时间监听
			clearInterval(intervalId)
            console.log('时间已到，开始抢购')
            cartGo()
        }
    } else {
        // 不在淘宝
        console.log('不在淘宝')
        // 停止时间监听
        clearInterval(intervalId)
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





