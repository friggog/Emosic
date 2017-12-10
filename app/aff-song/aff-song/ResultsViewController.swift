//
//  ResultsViewController.swift
//  aff-song
//
//  Created by Charlie Hewitt on 08/12/2017.
//  Copyright © 2017 Charlie Hewitt. All rights reserved.
//

import UIKit
import CoreML
import Vision

class ResultsViewController: AffUIViewController, UITableViewDataSource, UITableViewDelegate {
    
    var faceImage : UIImage?
    var tracks : [[String]]?
    @IBOutlet var imageView: UIImageView!
    @IBOutlet weak var TrackTable: UITableView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        imageView.image = faceImage
        let (valence, arousal) = getAffect(image: faceImage)
        getSongs(valence: valence, arousal: arousal, callback: {tracks in
            self.tracks = tracks
            DispatchQueue.main.async {
                self.TrackTable.reloadData()
            }
        })
    }
    
    @IBAction func backButtonClick(_ sender: Any) {
        dismiss(animated: true, completion: nil)
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    func getSongs(valence : Double, arousal: Double, callback: @escaping ([[String]])->Void){
        guard let authKey = UserDefaults.standard.string(forKey: "SpotifyAuthToken") else {
            fatalError("Spotify auth failed")
        }
        
        let urlPath: String = "https://api.spotify.com/v1/recommendations?seed_genres=pop,rock,dance&valence=\(valence)&danceability=\(arousal)&limit=5&market=GB"
        var request: URLRequest = URLRequest(url: URL(string: urlPath)!)
        request.httpMethod = "GET"
        request.addValue("Bearer " + authKey, forHTTPHeaderField: "Authorization")
        
        let session = URLSession.shared
        session.dataTask(with: request) {data, response, err in
            do {
                let jsonResult = try JSONSerialization.jsonObject(with: data!, options: []) as! [String:[Any]]
                var tracks : [[String]] = []
                for track in jsonResult["tracks"]! {
                    let track_d = track as! [String:Any]
                    let artists = track_d["artists"] as! [[String:Any]]
                    tracks.append([track_d["name"] as! String, artists[0]["name"] as! String, track_d["uri"] as! String])
                }
                callback(tracks)
            } catch let error {
                print(error.localizedDescription)
            }
            }.resume()
    }
    
    
    func getAffect(image: UIImage?) -> (Double, Double) {
        // TODO CNN
//        let model = nil
//        let pixelBuffer = UIImage(cgImage: image.cgImage).pixelBuffer()
//        guard let modelPrediction = try? model.prediction(image: pixelBuffer) else {
//            fatalError("Unexpected runtime error.")
//        }
        return (0.5, 0.5)
    }
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return self.tracks?.count ?? 0
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell:UITableViewCell = UITableViewCell(style:.subtitle, reuseIdentifier:"cell")
        cell.textLabel?.text = self.tracks![indexPath.row][0]
        cell.textLabel?.textColor = UIColor.white
        cell.detailTextLabel?.text = self.tracks![indexPath.row][1]
        cell.detailTextLabel?.textColor = UIColor.white
        cell.backgroundColor = .clear
        cell.selectionStyle = .none
        cell.backgroundColor = .clear
        return cell
    }
    
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        UIApplication.shared.openURL(URL(string: self.tracks![indexPath.row][2])!)
    }
}

extension UIImage {
    func pixelBuffer() -> CVPixelBuffer? {
        let width = self.size.width
        let height = self.size.height
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(width),
                                         Int(height),
                                         kCVPixelFormatType_OneComponent8,
                                         attrs,
                                         &pixelBuffer)
        
        guard let resultPixelBuffer = pixelBuffer, status == kCVReturnSuccess else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(resultPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(resultPixelBuffer)
        
        let grayColorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(data: pixelData,
                                      width: Int(width),
                                      height: Int(height),
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(resultPixelBuffer),
                                      space: grayColorSpace,
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
                                        return nil
        }
        
        context.translateBy(x: 0, y: height)
        context.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(resultPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return resultPixelBuffer
    }
}
