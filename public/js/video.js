const FPS = 60;
const faceCascadeFile = 'haarcascade_frontalface_default.xml'; // path to xml
const video = document.getElementById("cam_input"); // video is the id of video tag
const snapshot = document.getElementById("canvas_snapshot")
let context = snapshot.getContext("2d");

function openCvReady() {
  cv['onRuntimeInitialized']=()=>{
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then(function(stream) {
        video.srcObject = stream;
        video.play();
    })
    .catch(function(err) {
        console.log("An error occurred! " + err);
    });
    let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
    let gray = new cv.Mat();
    let cap = new cv.VideoCapture(cam_input);
    let faces = new cv.RectVector();
    let classifier = new cv.CascadeClassifier();
    let utils = new Utils('errorMessage');
    utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
    classifier.load(faceCascadeFile); // in the callback, load the cascade from file
});
    function processVideo() {
        let begin = Date.now();
        cap.read(src);
        src.copyTo(dst);
        cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
        get_faces();
        cv.imshow("canvas_output", dst);
        // schedule next one.
        let delay = 100/FPS - (Date.now() - begin);
        setTimeout(processVideo, delay);

    function get_faces(){
      try{
          classifier.detectMultiScale(gray, faces, 1.1, 3, 0);
          console.log(faces.size());
      }catch(err){
          console.log(err);
      }
      for (let i = 0; i < faces.size(); ++i) {
          let face = faces.get(i);
          let point1 = new cv.Point(face.x, face.y);
          let point2 = new cv.Point(face.x + face.width, face.y + face.height);
          cv.rectangle(dst, point1, point2, [255, 0, 0, 255]);
          take_snapshot(face);
      }
    }
    function take_snapshot(face){
      let rect = new cv.Rect(face.x, face.y, face.width, face.height)
      cropped = gray.roi(rect);
      cv.imshow("canvas_snapout", cropped);
    }
}
// schedule first one.
setTimeout(processVideo, 0);
  };
}
