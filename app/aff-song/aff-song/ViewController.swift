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

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate  {
    
    var spotifyAuthKey : String = ""
    var faceImage : UIImage?
    @IBOutlet weak var PhotoButton: UIButton!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        PhotoButton.layer.cornerRadius = PhotoButton.layer.frame.height/2
        let gradient = CAGradientLayer()
        gradient.frame = view.bounds
        let c1 = UIColor(red: 252.0/255.0, green: 49.0/255.0, blue: 89.0/255.0, alpha: 1.0)
        let c2 = UIColor(red: 252.0/255.0, green: 45.0/255.0, blue: 119.0/255.0, alpha: 1.0)
        gradient.colors = [c1.cgColor, c2.cgColor]
        view.layer.insertSublayer(gradient, at: 0)
        refreshAuth()
        // Do any additional setup after loading the view, typically from a nib.
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
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
        if let pickedImage = info[UIImagePickerControllerOriginalImage] as? UIImage {
            self.faceImage = detectFaces(image: pickedImage)!
            dismiss(animated: true, completion: nil)
            performSegue(withIdentifier: "resultsViewSegue", sender: self)
        }
        else {
            dismiss(animated: true, completion: nil)
        }
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.identifier == "resultsViewSegue" {
            if let imageViewController = segue.destination as? ResultsViewController {
                imageViewController.faceImage = self.faceImage
                imageViewController.spotifyAuthKey = self.spotifyAuthKey
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
    
    func refreshAuth() {
        // get a new auth key from spotify
        let url: String = "https://accounts.spotify.com/api/token"
        var request: URLRequest = URLRequest(url: URL(string: url)!)
        let bodyData = "grant_type=refresh_token&client_id=75f91608ce154091a1f419be415cbdda&client_secret=a6609b179eb140b086c2f6cc2c35adf4&refresh_token=AQBeDOqXW6kerokqh6WbEexkOQH5FtWGfDvw2DLePMofXTAVOUGsygF5iWYx0jtCjBUFO31qslCGNcRNh6vXGv9wxxbEuyNyWFy-t1YuYON2UD_ySlzSCcjAWsI2YiKzFAc"
        request.httpBody = bodyData.data(using: String.Encoding.utf8);
        request.httpMethod = "POST"

        let session = URLSession.shared
        session.dataTask(with: request) {data, response, err in
            do {
                let json = try JSONSerialization.jsonObject(with: data!, options: .allowFragments) as! [String:Any]
                if  let authKey = json["access_token"] {
                    self.spotifyAuthKey = authKey as! String
                }
            } catch let error {
                print(error.localizedDescription)
            }
            }.resume()
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

