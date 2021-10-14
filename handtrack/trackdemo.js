(function () {

    var clickDist = 55;
    var clickDist2 = 80;

    var width = 0;
    var height = 0;
    var ctx = null;

    var dragging = false;

    var allPath = [];
    var curPath = [];

    function point(lm) {
        return {x: lm.x * width, y: lm.y * height}
    }

    function dist(p1, p2) {
        var dx = p1.x - p2.x;
        var dy = p1.y - p2.y;
        return Math.sqrt(dx* dx, dy* dy);
    }

    function endPath(pt) {
        console.log("end");
        curPath.push(pt);
        if (curPath.length > 1) {
            allPath.push(curPath);
        }
        curPath = [];
    }

    function startPath(pt) {
        console.log("start");
        curPath = [pt];
    }

    function updatePath(pt) {
        curPath.push(pt);
    }

    function checkClick(p8, p4, p10) {
        var d = dist(p4, p10);
        if (dragging) {
            if (d > clickDist2) {
                endPath(p8);
                dragging = false;
            }
        } else {
            if (d < clickDist) {
                startPath(p8);
                dragging = true;
            }
        }
    }

    window.init_tracker = function(canvasElement) {
        width = canvasElement.width;
        height = canvasElement.height;
        ctx = canvasElement.getContext('2d');
    }

    window.handleLandmarks = function(landmarks) {
        var p8 = point(landmarks[8]);
        var p4 = point(landmarks[4]);
        var p10 = point(landmarks[10]);
        checkClick(p8, p4, p10);
        if (dragging) {
            updatePath(p8);
        }
    }

    window.draw_tracker = function() {
        function dp(path) {
            if (path.length < 2) {
                return;
            }
    
            ctx.beginPath();
            ctx.moveTo(path[0].x, path[0].y);
            for (let i = 1; i < path.length; i++) {
                //debugger
                ctx.lineTo(path[i].x, path[i].y);
            }
            ctx.stroke();
        }

        dp(curPath);
        for (let p of allPath) {
            dp(p);
        }
    }

})();