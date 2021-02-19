<template>
	<view class="form">
		<view class="cu-form-group">
			<view class="title">Host</view>
			<input placeholder="127.0.0.1" name="host" type="text"  v-model="host" @change="sD" />
		</view>
		<view class="cu-form-group">
			<view class="title">Port</view>
			<input placeholder="8000" name="port" type="number"  @change="sD" v-model="port" />
		</view>
	</view>
</template>

<script>
var data1 = {};

data1['port'] = '8000';

data1['host'] = '127.0.0.1';

uni.getStorage({
	key: 'port',
	success: function(res) {
		console.log(res);
		data1['port'] = res.data;
	}
});
uni.getStorage({
	key: 'host',
	success: function(res) {
		data1['host'] = res.data;
	}
});

export default {
	data() {
		return data1;
	},
	methods: {
		sD: function(e) {
			uni.setStorage({
				key: 'host',
				data: this.host
			});
			uni.setStorage({
				key: 'port',
				data: this.port
			});
		},
	}
};
</script>

<style>
.cu-form-group .title {
	min-width: calc(4em + 15px);
}
.form {
	margin-top: 120px;
}
</style>
