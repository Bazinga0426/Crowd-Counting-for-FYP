<template>
	<view class="justify-center" style="text-align: center;" >
		<cu-custom bgColor="bg-gradual-blue" :isBack="true">
			<block slot="backText">back</block>
			<block slot="content">Crowd Counting</block>
		</cu-custom>
		<image :src="img" v-model="img" mode="widthFix" class="response"></image>
		<hr  v-if="A_SUM!=''">
		<br>
		<br>
		<br>
		<view class=" justify-center" style="margin: 0 auto;font-size: 20px;"   v-if="A_SUM!=''">
			Model A predict Heat map:
			<image :src="imgA" v-model="imgA" v-if="A_SUM!=''" mode="widthFix" class="response"></image>
			</view>
		<view class="justify-center" style="margin: 0 auto;font-size: 20px;"  v-if="A_SUM!=''">Model A predict: <strong> {{A_SUM}}</strong>People</view>
		<hr  v-if="A_SUM!=''">
		<br>
		<br>
		<br>
		<view class=" justify-center" style="margin: 0 auto;font-size: 20px;"  v-if="A_SUM!=''">
		Model B predict Heat map:	
		<image :src="imgB" v-model="imgB" v-if="B_SUM!=''" mode="widthFix" class="response"></image>
		</view>
		<view class=" justify-center" style="margin: 0 auto;font-size: 20px;"  v-if="B_SUM!=''">Model B predict: <strong>{{B_SUM}}</strong>People</view>
		<br>
		<br>
		<br>
		<br>
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
	onLoad: function(option) {
		console.log('onLoad', option.img);
		data1['img'] = option.img;
		this.img = option.img;
		this.upload_url= 'http://' + data1['host'] + ':' + data1['port']
	},
	onShow: function() {
		console.log('开始上传', this.upload_url, this.img);
		var vue_this = this;
		uni.uploadFile({
			url: this.upload_url,
			fileType: 'image',
			filePath: this.img,
			name: 'img',
		
			fail: function(error) {
				console.log(error);
			},
			success: function(uploadFileRes) {
			var 	data = JSON.parse(uploadFileRes.data);
			vue_this.A_SUM = data.A_SUM;
			vue_this.B_SUM = data.B_SUM;
			vue_this.imgA = data.A;
			vue_this.imgB = data.B;
			}
		});
	},
	data() {
		return {
			img: data1['img'],
			upload_url: 'http://' + data1['host'] + ':' + data1['port'],
			A_SUM:"","B_SUM":"","imgA":"","imgB":""
		};
	},
	methods: {}
};
</script>

<style>
page {
	padding-top: 50px;
}
</style>
