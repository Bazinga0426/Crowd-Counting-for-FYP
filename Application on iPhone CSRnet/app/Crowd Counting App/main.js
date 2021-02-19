import Vue from 'vue'
import App from './App'

import components from './pages/component/home.vue'
Vue.component('components',components)

import settings from './pages/settings/settings.vue'
Vue.component('settings',settings)

import cuCustom from './colorui/components/cu-custom.vue'
Vue.component('cu-custom',cuCustom)

Vue.config.productionTip = false

App.mpType = 'app'

const app = new Vue({
    ...App
})
app.$mount()

 



