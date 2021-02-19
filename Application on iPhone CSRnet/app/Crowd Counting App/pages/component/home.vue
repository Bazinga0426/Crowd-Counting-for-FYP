<template name="components">
	<view>
		<scroll-view scroll-y class="page">
			<image src="/static/componentBg.png" mode="widthFix" class="response"></image>
			<view class="nav-list">
				<navigator
					@click="NavChange"
					:data-url="item.url"
					:id="item.id"
					hover-class="none"
					class="nav-li"
					style="width:100%;"
					navigateTo
					:class="'bg-' + item.color"
					:style="[{ animation: 'show ' + ((index + 1) * 0.2 + 1) + 's 1' }]"
					v-for="(item, index) in elements"
					:key="index"
				>
					<view class="nav-title">{{ item.title }}</view>
					<view class="nav-name">{{ item.name }}</view>
					<text :class="'cuIcon-' + item.cuIcon"></text>
				</navigator>
			</view>
			<view class="cu-tabbar-height"></view>
			<view class="cu-load load-modal" v-if="loadModal">
				<!-- <view class="cuIcon-emojifill text-orange"></view> -->

				<view class="gray-text">Servers Predicting...</view>
			</view>
		</scroll-view>
	</view>
</template>

<script>


export default {
	data() {
		return {
			elements: [
				{
					title: '',
					name: 'photograph',
					color: 'purple',
					cuIcon: 'lg text-gray cuIcon-cameraadd'
				},
				{
					title: ' ',
					name: 'Select Image',
					id: 'Select-Image',
					color: 'mauve',
					cuIcon: ' cuIcon-pic'
				}
			],
			loadModal: false,
			
			
		};
	},
	methods: {
		NavChange: function(e) {
			console.log(e);
			if (e.currentTarget.id == 'Select-Image') {
				this.select_image(e);
			} else {
				this.camera_image(e);
			}

			console.log(e);
		},
		select_image: function(e) {
			var vue_this = this;
			uni.chooseImage({
				count: 1,
				sizeType: ['original'],
				sourceType: ['album'],
				success: function(res) {
					vue_this.urlTobase64(res.tempFilePaths[0], vue_this);
		
				}
			});
		},
		camera_image: function(e) {
			var vue_this = this;
			console.log('camera');
			uni.chooseImage({
				count: 1,
				sizeType: ['original'],
				sourceType: ['camera'],
				success: function(res) {
					vue_this.urlTobase64(res.tempFilePaths[0], vue_this);

			
				}
			});
		},
		urlTobase64(url, vue_this) {
			vue_this.png = url;
			uni.navigateTo({
			
			url: '/pages/camera/upload?img='+url
			});
		
			return
			


		},
		
		

	
	}
};


</script>

<style>
.page {
	height: 100vh;
	background-image: rgb(156, 38, 176);
}
.nav-list {
	position: absolute;
	width: 100%;
	top: 30vh;
}
</style>
