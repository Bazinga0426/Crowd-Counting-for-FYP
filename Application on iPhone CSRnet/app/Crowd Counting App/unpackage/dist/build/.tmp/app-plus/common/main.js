(global["webpackJsonp"]=global["webpackJsonp"]||[]).push([["common/main"],{"1dff":function(t,e,n){"use strict";(function(t){Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0;var n={data:function(){var e={};return window.localStorage.getItem("port")?e["port"]=window.localStorage.getItem("port"):e["port"]="8000",window.localStorage.getItem("host")?e["host"]=window.localStorage.getItem("host"):e["host"]="127.0.0.1",t("log",e," at pages\\settings\\settings.vue:32"),e},methods:{setData:function(t){window.localStorage.host=this.host,window.localStorage.port=this.port}}};e.default=n}).call(this,n("0de9")["default"])},3184:function(t,e,n){"use strict";var o,r=function(){var t=this,e=t.$createElement;t._self._c},a=[];n.d(e,"b",(function(){return r})),n.d(e,"c",(function(){return a})),n.d(e,"a",(function(){return o}))},"5d88":function(t,e,n){"use strict";(function(t){n("ca3c"),n("921b");var e=a(n("66fd")),o=a(n("ea4d")),r=a(n("d6c9"));function a(t){return t&&t.__esModule?t:{default:t}}function c(t,e){var n=Object.keys(t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(t);e&&(o=o.filter((function(e){return Object.getOwnPropertyDescriptor(t,e).enumerable}))),n.push.apply(n,o)}return n}function u(t){for(var e=1;e<arguments.length;e++){var n=null!=arguments[e]?arguments[e]:{};e%2?c(Object(n),!0).forEach((function(e){l(t,e,n[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(n)):c(Object(n)).forEach((function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(n,e))}))}return t}function l(t,e,n){return e in t?Object.defineProperty(t,e,{value:n,enumerable:!0,configurable:!0,writable:!0}):t[e]=n,t}var i=function(){return n.e("pages/component/home").then(n.bind(null,"c8cf"))};e.default.component("components",i),e.default.component("settings",r.default);var f=function(){return n.e("colorui/components/cu-custom").then(n.bind(null,"8fc9"))};e.default.component("cu-custom",f),e.default.config.productionTip=!1,o.default.mpType="app";var d=new e.default(u({},o.default));t(d).$mount()}).call(this,n("6e42")["createApp"])},"7f5c":function(t,e,n){"use strict";var o=n("909b"),r=n.n(o);r.a},"909b":function(t,e,n){},"95ac":function(t,e,n){"use strict";n.r(e);var o=n("1dff"),r=n.n(o);for(var a in o)"default"!==a&&function(t){n.d(e,t,(function(){return o[t]}))}(a);e["default"]=r.a},bae6:function(t,e,n){"use strict";n.r(e);var o=n("cfd0"),r=n.n(o);for(var a in o)"default"!==a&&function(t){n.d(e,t,(function(){return o[t]}))}(a);e["default"]=r.a},cfd0:function(t,e,n){"use strict";(function(t,o){Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0;var r=a(n("66fd"));function a(t){return t&&t.__esModule?t:{default:t}}var c={onLaunch:function(){t.getSystemInfo({success:function(t){r.default.prototype.StatusBar=t.statusBarHeight,"android"==t.platform?r.default.prototype.CustomBar=t.statusBarHeight+50:r.default.prototype.CustomBar=t.statusBarHeight+45}}),r.default.prototype.ColorList=[{title:"嫣红",name:"red",color:"#e54d42"},{title:"桔橙",name:"orange",color:"#f37b1d"},{title:"明黄",name:"yellow",color:"#fbbd08"},{title:"橄榄",name:"olive",color:"#8dc63f"},{title:"森绿",name:"green",color:"#39b54a"},{title:"天青",name:"cyan",color:"#1cbbb4"},{title:"海蓝",name:"blue",color:"#0081ff"},{title:"姹紫",name:"purple",color:"#6739b6"},{title:"木槿",name:"mauve",color:"#9c26b0"},{title:"桃粉",name:"pink",color:"#e03997"},{title:"棕褐",name:"brown",color:"#a5673f"},{title:"玄灰",name:"grey",color:"#8799a3"},{title:"草灰",name:"gray",color:"#aaaaaa"},{title:"墨黑",name:"black",color:"#333333"},{title:"雅白",name:"white",color:"#ffffff"}]},onShow:function(){o("log","App Show"," at App.vue:109")},onHide:function(){o("log","App Hide"," at App.vue:112")}};e.default=c}).call(this,n("6e42")["default"],n("0de9")["default"])},d6c9:function(t,e,n){"use strict";n.r(e);var o=n("3184"),r=n("95ac");for(var a in r)"default"!==a&&function(t){n.d(e,t,(function(){return r[t]}))}(a);n("e3db");var c,u=n("f0c5"),l=Object(u["a"])(r["default"],o["b"],o["c"],!1,null,null,null,!1,o["a"],c);e["default"]=l.exports},e3db:function(t,e,n){"use strict";var o=n("f48d"),r=n.n(o);r.a},ea4d:function(t,e,n){"use strict";n.r(e);var o=n("bae6");for(var r in o)"default"!==r&&function(t){n.d(e,t,(function(){return o[t]}))}(r);n("7f5c");var a,c,u,l,i=n("f0c5"),f=Object(i["a"])(o["default"],a,c,!1,null,null,null,!1,u,l);e["default"]=f.exports},f48d:function(t,e,n){}},[["5d88","common/runtime","common/vendor"]]]);