//
//  USEmojiViewController.swift
//  aff-song
//
//  Created by Charlie Hewitt on 10/12/2017.
//  Copyright © 2017 Charlie Hewitt. All rights reserved.
//

import Foundation
import Vision

class USEmojiViewController : AffUIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate  {
    @IBOutlet weak var EmotionLabel: UILabel!
    @IBOutlet weak var EmojiLabel: UILabel!
    
    var emotion : Int = 0
    var emotionLabel : Int = 0
    var faceImage : UIImage?
    var data : [String] = []
    
    @IBAction func unwindToEmoji(unwindSegue: UIStoryboardSegue) {
        emotion += 1
        updateForEmotionChange()
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        updateForEmotionChange()
    }

    func updateForEmotionChange() {
        var emotionText : String?
        var emotionEmoji : String?
        switch emotion {
        case 0:
            emotionText = "Neutral"
            emotionEmoji = "😐"
            emotionLabel = 0
            break
        case 1:
            emotionText = "Delighted"
            emotionEmoji = "😀"
            emotionLabel = 1
            break
        case 2:
            emotionText = "Happy"
            emotionEmoji = "🙂"
            emotionLabel = 1
            break
        case 3:
            emotionText = "Miserable"
            emotionEmoji = "😞"
            emotionLabel = 2
            break
        case 4:
            emotionText = "Sad"
            emotionEmoji = "☹️"
            emotionLabel = 2
            break
        case 5:
            emotionText = "Suprised"
            emotionEmoji = "😮"
            emotionLabel = 3
            break
        case 6:
            emotionText = "Angry"
            emotionEmoji = "😡"
            emotionLabel = 6
            break
        case 7:
            emotionText = "Afraid"
            emotionEmoji = "😰"
            emotionLabel = 4
            break
        case 8:
            emotionText = "Disgusted"
            emotionEmoji = "😬"
            emotionLabel = 5
            break
        case 9:
            emotionText = "Contemptful"
            emotionEmoji = "🤨"
            emotionLabel = 7
            break
        default:
            emotionText = "Error"
            emotionEmoji = "🤪"
        }
        EmotionLabel.text = emotionText
        EmojiLabel.text = emotionEmoji
    }
    
    @IBAction func openCameraButton(sender: AnyObject) {
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            let imagePicker = UIImagePickerController()
            imagePicker.delegate = self
            imagePicker.sourceType = .camera
            imagePicker.cameraDevice = .front
            imagePicker.allowsEditing = false
            self.present(imagePicker, animated: true, completion: nil)
        }
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        if let pickedImage = info[UIImagePickerControllerOriginalImage] as? UIImage,
            let face = detectFaces(image: pickedImage) {
            faceImage = face
            dismiss(animated: true, completion: {
                self.performSegue(withIdentifier: "usResultsSegue", sender: self)
            })
        }
        else {
            dismiss(animated: true, completion: nil)
        }
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.identifier == "usResultsSegue" {
            if let nextViewController = segue.destination as? USResultsViewController {
                nextViewController.faceImage = self.faceImage
                nextViewController.emotion = self.emotion
                nextViewController.emotionLabel = self.emotionLabel
                nextViewController.data = self.data
            }
        }
    }
    
    func detectFaces(image: UIImage) -> UIImage? {
        let orientation = CGImagePropertyOrientation(rawValue: imageOrientationToExif(image: image))!
        let faceDetectionRequest = VNDetectFaceRectanglesRequest()
        let myRequestHandler = VNImageRequestHandler(cgImage: image.cgImage!, orientation: orientation, options: [:])
        try! myRequestHandler.perform([faceDetectionRequest])
        guard let results = faceDetectionRequest.results,
            let cgImage = image.cgImage,
            results.count > 0 else {
                return nil
        }
        for observation in faceDetectionRequest.results as! [VNFaceObservation] {
            let box = observation.boundingBox
            // iOS is a joke for image handling...
            // flip h/w as in portrait but cg image is landscape
            let imw = CGFloat(cgImage.height)
            let imh = CGFloat(cgImage.width)
            let w = box.size.width * imw
            let h = box.size.height * imh
            let x = box.origin.x * imw
            let y = box.origin.y * imh
            // and fix for stupid coordinate system inconsistencies
            let cropRect = CGRect(x: imh - (y+h), y: imw - (x+w), width: h, height: w)
            let croppedCGImage: CGImage = cgImage.cropping(to: cropRect)!
            let croppedUIImage: UIImage = UIImage(cgImage: croppedCGImage, scale: image.scale, orientation: image.imageOrientation)
            let size = CGSize(width:256,height:256)
            UIGraphicsBeginImageContextWithOptions(size, false, image.scale)
            croppedUIImage.draw(in: CGRect(origin: CGPoint.zero, size: size))
            let scaledImage = UIGraphicsGetImageFromCurrentImageContext()!
            UIGraphicsEndImageContext()
            return scaledImage
        }
        return nil
    }
    
    func imageOrientationToExif(image: UIImage) -> uint {
        switch image.imageOrientation {
        case .up:
            return 1;
        case .down:
            return 3;
        case .left:
            return 8;
        case .right:
            return 6;
        case .upMirrored:
            return 2;
        case .downMirrored:
            return 4;
        case .leftMirrored:
            return 5;
        case .rightMirrored:
            return 7;
        }
    }
    
    @IBAction func cancelButtonClicked(_ sender: Any) {
        let alert = UIAlertController(title: "Exit User Study", message: "Are you sure you would like to exit the study?", preferredStyle: UIAlertControllerStyle.alert)
        alert.addAction(UIAlertAction(title: "Exit", style: UIAlertActionStyle.destructive, handler: { action in
            self.navigationController?.dismiss(animated: true, completion: nil)
        }))
        alert.addAction(UIAlertAction(title: "Cancel", style: UIAlertActionStyle.cancel, handler: nil))
        self.present(alert, animated: true, completion: nil)
    }
}
