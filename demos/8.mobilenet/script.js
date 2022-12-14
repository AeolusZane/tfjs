// 1个 RGBA RGBA(data[0],data[1],data[2],data[3])
// 2个 RGBA RGBA(data[4],data[5],data[6],data[7])
// 第N个像素信息为:
//  RGBA(data[(n-1)*4],data[(n-1)*4+1],data[(n-1)*4+2],data[(n-1)*4+3])
// 200*200 每1个像素1行1列 200行 200列 i行 j列
// pos = [(i-1)*200,[j-1]]*4;

var canvas = document.getElementById('myCanvas');
var ctx = canvas.getContext('2d');

var image = new Image();
image.src = 'star.png';

var pixels = []; //存储像素数据
var imageData;
image.onload = function () {
  ctx.drawImage(image, 200, 100, 200, 200);
  imageData = ctx.getImageData(200, 100, 200, 200); //获取图表像素信息
  getPixels(); //获取所有像素
  drawPic(); //绘制图像
};

function getPixels() {
  var pos = 0;
  var data = imageData.data; //RGBA的一维数组数据
  console.log('data: ', data);
  //源图像的高度和宽度为200px
  for (var i = 1; i <= 200; i++) {
    for (var j = 1; j <= 200; j++) {
      pos = [(i - 1) * 200 + (j - 1)] * 4; //取得像素位置
      if (data[pos] >= 0) {
        var pixel = {
          x: 200 + j + Math.random() * 20, //重新设置每个像素的位置信息
          y: 100 + i + Math.random() * 20, //重新设置每个像素的位置信息
          fillStyle:
            'rgba(' +
            data[pos] +
            ',' +
            data[pos + 1] +
            ',' +
            data[pos + 2] +
            ',' +
            data[pos + 3] +
            ')',
        };
        pixels.push(pixel);
      }
    }
  }
}

function drawPic() {
  var canvas = document.getElementById('myCanvas');
  var ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, 600, 400);
  var len = pixels.length,
    curr_pixel = null;
  for (var i = 0; i < len; i++) {
    curr_pixel = pixels[i];
    ctx.fillStyle = curr_pixel.fillStyle;
    ctx.fillRect(curr_pixel.x, curr_pixel.y, 1, 1);
  }
}
