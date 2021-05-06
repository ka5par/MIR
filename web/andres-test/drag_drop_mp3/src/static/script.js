var audioContext = null;

function touchStarted() {
  audioContext = new AudioContext().resume()
}

var rects = {};
var canvas;
var waveforms = new Array();
var WAVEFORM;
var oAudio = '';
var cc = 1;
var view1 = $('#view1');
var view2 = $('#view2');
var view3 = $('#view3');
var v2_offset = view2.offset();

(function (c, cx) {
  $(document).on('dragover', function(){
    $('#__drop').addClass('show');
    return false;
  });
  
  $('#__drop').on('drop', function(e){
    e.stopPropagation();
    e.preventDefault();
    var data = e.originalEvent.dataTransfer;
    file = data.files[0];
    var file_name = file.name.substring(0, file.name.length - 4);
    
    $.when(initAudio(data)).done(function (b) {
        clearCanvas();
        setupBars(b);
        $('#music_title').html(file_name);
		    $('#audio')[0].src = URL.createObjectURL(file);
    });
    
    $('#__drop').removeClass('show').addClass('hidden');
  });
  
  $('#__drop').on('dragleave', function(){
    $('#__drop').removeClass('show').addClass('hidden');
  });
  
    oAudio = document.getElementById('audio');
    oAudio.addEventListener("timeupdate", progressBar, true);
  
    window.WAVEFORM = WAVEFORM = function (cx, x, y, w, h, speed) {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
		this.ctx = cx;
		
		this.trigger = false;
		this.alpha = 0;
		this.speed = speed;
		this.done = false;
    }

    WAVEFORM.prototype = {
      redraw: function(){
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.clearRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);
        this.ctx.restore();
      },
      isPointBar: function(x, y){
        return (x >= (this.x * 3) && x <= (this.x * 3) + this.w);
      },
      highlight: function(){
        var c3 = document.getElementById('view3');
        var _ctx = c3.getContext('2d');
        _ctx.setTransform(1, 0, 0, 1, 0, 0);
        _ctx.clearRect(0,0, c.width, c.height);
        this.fillBars();
      },
      fillBars: function(){
        var barX = (this.x * 3) + this.w;
        var c3 = document.getElementById('view3');
        var _ctx = c3.getContext('2d');
        
        for(var i = 0; i < barX/3; i++) {
          _ctx.translate(0, c.height / 2);
          _ctx.scale(1, -1);
          _ctx.fillStyle = "#ff894d";//"#FF5600";
          _ctx.fillRect(waveforms[i].x * 3, waveforms[i].y, waveforms[i].w, waveforms[i].h);
          _ctx.setTransform(1, 0, 0, 1, 0, 0);
          _ctx.fillStyle = "#f8e5d9";//"#F0C7AE";
          _ctx.fillRect(waveforms[i].x * 3, c.height / 2 + 11, waveforms[i].w, waveforms[i].h / 2);
        }
      },
		displayBar : function (x) {
			var _this = this;
			var speed = this.speed;
			_this.ctx.save();
			var fadeIn = function(){
				_this.ctx.translate(0, c.height / 2);
				_this.ctx.scale(1, -1);
				_this.ctx.fillStyle = "rgba(255, 86, 0, "+ speed +")";//"#FF5600";
				_this.ctx.fillRect(_this.x * 3, _this.y, _this.w, _this.h);
				_this.ctx.setTransform(1, 0, 0, 1, 0, 0);
				_this.ctx.fillStyle = "rgba(240, 199, 174, "+ speed +")";//"#F0C7AE";
				_this.ctx.fillRect(_this.x * 3, c.height / 2 + 11, _this.w, _this.h / 2);
				
				speed += speed;
				
				var fade = requestAnimationFrame(fadeIn);
				
				if(speed > 1) {
					cancelAnimationFrame(fade);
				}
			}
			
			fadeIn();
		},
		trigger: function(){
			this.trigger.true;
		}
	}
    var j = 0;
    function progressBar() {
        var oAudio = document.getElementById('audio');
        var elapsedTime = Math.round(oAudio.currentTime);
        var fWidth = Math.floor((elapsedTime / oAudio.duration) * (c.width));
        var p = Math.ceil(fWidth/3);

        if (!oAudio.paused && p > 0) {
			for(j = 0; j < p; j++) {
				if (typeof waveforms[j] != 'undefined') {
					waveforms[j].displayBar();
				}
			}
			j = Math.max(j + 1, p + 1);
        }
    }

    var setupBars = function (b) {
        var data = b.getChannelData(0);
        var step = Math.ceil(data.length / c.width);
        var amp = (c.height / 2);
        var oAudio = document.getElementById('audio');
        var c2 = document.getElementById('view2');
        var ctx = c2.getContext('2d');
      
        for (var i = 0; i < c.width; i++) {
            var min = 1.0;
            var max = -1.0;

            for (var j = 0; j < step; j++) {
                var datum = data[(i * step * 3) + j];
                if (datum > max)
                    max = datum;
            }

            cx.translate(0, c.height / 2);
            cx.scale(1, -1);

            cx.fillStyle = "#E5E5E5";
            cx.fillRect(i * 3, -10, 2, max * amp);
            cx.setTransform(1, 0, 0, 1, 0, 0);
            cx.fillStyle = "#9DA09B";
            cx.fillRect(i * 3, c.height / 2 + 12, 2, max * amp / 2);
            
            window.waveforms[i] = waveforms[i] = new WAVEFORM(ctx, i, -10, 2, max * amp, 0.02);
        }
    }
}(document.getElementById('view1'), document.getElementById('view1').getContext('2d')));

function initAudio(data) {
    var audioRequest = new XMLHttpRequest();
    var dfd = jQuery.Deferred();
    
    audioRequest.open("GET", URL.createObjectURL(data.files[0]), true);
    audioRequest.responseType = "arraybuffer";
    audioRequest.onload = function () {
        audioContext.decodeAudioData(audioRequest.response,
                function (buffer) {
                    dfd.resolve(buffer);
                });
    }
    audioRequest.send();

    return dfd.promise();
}

view3.on('mouseout', function(){
  $('#view2').removeClass('fadeOut').addClass('fadeIn');
  var c3 = document.getElementById('view3');
  var _ctx = c3.getContext('2d');
  _ctx.setTransform(1, 0, 0, 1, 0, 0);
  _ctx.clearRect(0,0, c3.width, c3.height);  
});

view3.on('mousemove', function(e){
    mouseX = parseInt(e.clientX - v2_offset.left);
    mouseY = parseInt(e.clientY - v2_offset.top);
    
    $('#view2').removeClass('fadeIn').addClass('fadeOut');
    // Put your mousemove stuff here
    //ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (var i = 0; i < waveforms.length; i++) {
      if (waveforms[i].isPointBar(mouseX, mouseY)) {
        waveforms[i].highlight();
      } else {
        //waveforms[i].redraw();
      }
    }
});

view3.on('click', function(e){
    mouseX = parseInt(e.clientX - v2_offset.left);
    mouseY = parseInt(e.clientY - v2_offset.top);

    // Put your mousemove stuff here
    //ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (var i = 0; i < waveforms.length; i++) {
      if (waveforms[i].isPointBar(mouseX, mouseY)) {
        waveforms[i].highlight();
        var percent = oAudio.duration * mouseX;
        oAudio.currentTime = percent/waveforms[i].ctx.canvas.width;
      } else {
        waveforms[i].redraw();
      }
    }
});

function clearCanvas(){
  var canvas1 = document.getElementById('view1');
  var ctx1 = canvas1.getContext('2d');
  
  var canvas2 = document.getElementById('view2');
  var ctx2 = canvas2.getContext('2d');
  
  ctx1.setTransform(1,0,0,1,0,0);
  ctx2.setTransform(1,0,0,1,0,0);
  
  ctx1.clearRect(0, 0, canvas1.width, canvas1.height);
  ctx2.clearRect(0, 0, canvas2.width, canvas2.height);
}