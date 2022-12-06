document.addEventListener("DOMContentLoaded", function () {
    suggest();
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#FFFFFF";

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    function draw(e) {
        if (!isDrawing) return;

        console.log(e);
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }

    canvas.addEventListener("mousedown", (e) => {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    });
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", () => isDrawing = false);
    canvas.addEventListener("mouseout", () => isDrawing = false);
});

function submitImage() {
    const canvas = document.getElementById("canvas");
    // var img = canvas.toDataURL();
    var img = canvas.toDataURL("image/png").replace("image/png", "image/octet-stream");  // here is the most important part because if you dont replace you will get a DOM 18 exception.
    // window.location.href=image;
    // console.log("SUBMIT");
    // document.getElementById("image")=img;
    var link = document.getElementById('image');
    link.setAttribute('value', canvas.toDataURL("image/png").replace("image/png", "image/octet-stream"));
    // var dataURL = canvas.toDataURL("image/jpeg", 1.0);
    // downloadImage(img, 'my-canvas.jpeg');
}
// Save | Download image
function downloadImage(data, filename = 'untitled.jpeg') {
    var a = document.createElement('a');
    a.href = data;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
}
function predictImage(img) {
    fetch("/predict", {
        method: "POST",
        body: img
    }).then(resp => resp.text())
    .then(data => {
        const myguess = document.getElementById("myguess");
        myguess.textContent = "My guess:"

        console.log(data);
        var strings = data.split(" ");
        var character = strings[0];
        var prob = strings[1];

        const guess = document.getElementById("guess");
        guess.textContent = character;

        const confidence = document.getElementById("confidence");
        confidence.textContent = "(confidence: " + prob + "%)";
    });
}

function clearCanvas() {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function suggest() {
    const suggestion = document.getElementById("suggestion");
    fetch("/suggest").then(response => response.text())
    .then(data => suggestion.textContent = "Draw a Tamil character. Not sure what to draw? Try " + data + ".");
}
function save() {
    var image = new Image();
    var canvas = document.getElementById("canvas");
    var url = document.getElementById('url');
    image.id = "pic";
    image.src = canvas.toDataURL("image/png").replace("image/png", "image/octet-stream"); ;
    url.value = image.src;
    console.log(url.value)
    console.log("save function triggered")
    // var t=prompt();

    document.getElementById("ceg").submit();
}