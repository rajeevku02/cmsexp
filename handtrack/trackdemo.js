(function () {
    const clickDist = 55;
    const clickDist2 = 80;
    const minPointDist = 3;
    const dragMinDist = 12;

    var width = 0;
    var height = 0;
    var ctx = null;

    class AppInput {
        constructor() {
            this.path = new PathPoints();
        }

        click(x, y) {
            console.log("click: ", x, y);
        }

        drag(x, y, kind) {
            if (kind == 0) {
                this.path.begin();
            }
            this.path.addPoint({x:x, y:y});
            if (kind == 2) {
                this.path.finish();
            }
        }

        move(x, y) {
        }

        draw(ctx) {
            this.path.draw(ctx);
        }
    }

    class PathPoints {
        constructor() {
            this.points = [];
        }

        begin() {
            this.points = [];
        }

        finish() {
        }

        addPoint(pt) {
            if (this.points.length == 0) {
                this.points.push(pt);
                return;
            }
            var d = dist(this.points[this.points.length - 1], pt);
            if (d > minPointDist) {
                this.points.push(pt);
            }
        }

        draw(ctx) {
            var len = this.points.length;
            if (len < 2) {
                return;
            }
            ctx.beginPath();
            ctx.moveTo(this.points[0].x, this.points[0].y);
            for (let i = 1; i < len; i++) {
                ctx.lineTo(this.points[i].x, this.points[i].y);
            }
            ctx.stroke();
        }
    }

    class HandInput {
        constructor() {
            this.mdown = false;
            this.moved = false;
            this.initialX = null;
            this.initialY = null;

            this.dragging = false;
        }

        down(x, y) {
            //console.log("down")
            this.initialX = x;
            this.initialY = y;
            this.mdown = true;
            this.moved = false;
        }

        up(x, y) {
            //console.log("up")
            if (this.moved) {
                appInput.drag(x, y, 2);
            } else if (this.mdown) {
                appInput.click(x, y);
            }
            this.mdown = false;
            this.moved = false;
        }

        move(x, y) {
            if (!this.mdown) {
                appInput.move(x, y);
                return;
            }
            if (this.moved) {
                appInput.drag(x, y, 1);
            } else {
                var d = dist({x: this.initialX, y: this.initialY}, {x: x, y:y});
                if (d >= dragMinDist) {
                    this.moved = true;
                    appInput.drag(this.initialX, this.initialY, 0);
                    appInput.drag(x, y, 1);
                }
            }
        }

        checkDown(p8, p4, p10) {
            var d = dist(p4, p10);
            if (this.dragging) {
                if (d > clickDist2) {
                    this.up(p8.x, p8.y);
                    this.dragging = false;
                }
            } else {
                if (d < clickDist) {
                    this.down(p8.x, p8.y);
                    this.dragging = true;
                }
            }
        }

        handleLandmarks(p8, p4, p10) {
            this.checkDown(p8, p4, p10);
            this.move(p8.x, p8.y);
        }
    }

    const appInput = new AppInput();
    const handInput = new HandInput();

    function point(lm) {
        return {x: lm.x * width, y: lm.y * height}
    }

    function dist(p1, p2) {
        var dx = p1.x - p2.x;
        var dy = p1.y - p2.y;
        return Math.sqrt(dx* dx, dy* dy);
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
        handInput.handleLandmarks(p8, p4, p10);
    }

    window.draw_tracker = function() {
        appInput.draw(ctx);
    }

})();
