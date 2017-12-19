//
//  ViewController.swift
//  aff-song
//
//  Created by Charlie Hewitt on 08/12/2017.
//  Copyright © 2017 Charlie Hewitt. All rights reserved.
//

import UIKit
import Vision
import CoreGraphics

class ViewController: AffUIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate  {
    
    var faceImage : UIImage?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
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
            self.faceImage = face
            dismiss(animated: true, completion: {
                self.performSegue(withIdentifier: "resultsViewSegue", sender: self)
            })
        }
        else {
            let alert = UIAlertController(title: "No Face Detected", message: "Please ensure your face is clearly visible in the image.", preferredStyle: UIAlertControllerStyle.alert)
            alert.addAction(UIAlertAction(title: "OK", style: UIAlertActionStyle.cancel, handler: nil))
            self.dismiss(animated: true, completion: {
                self.present(alert, animated: true, completion: nil)
            })
        }
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.identifier == "resultsViewSegue" {
            if let imageViewController = segue.destination as? ResultsViewController {
                imageViewController.faceImage = self.faceImage
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
            let size = CGSize(width:310, height:310)
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
}

