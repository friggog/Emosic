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
        let (valence, arousal, emotion) = getAffect(image: faceImage)
        getSongs(valence: valence, arousal: arousal, emotion: emotion, callback: {tracks in
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
    
    func getSongs(valence : Double, arousal: Double, emotion: Int, callback: @escaping ([[String]])->Void){
        guard let authKey = UserDefaults.standard.string(forKey: "SpotifyAuthToken") else {
            fatalError("Spotify auth failed")
        }
        
        // TODO map emotion to genres
        let genres = "rock,pop,dance"
        let urlPath: String = "https://api.spotify.com/v1/recommendations?seed_genres=\(genres)&valence=\(valence)&energy=\(arousal)&limit=5&market=GB"
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
    
    
    func getAffect(image: UIImage?) -> (Double, Double, Int) {
        // 0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt, 8: None, 9: Uncertain, 10: No-Face
        // TODO CNN predicted
        let pixelBuffer = image?.pixelBuffer(width: 96, height: 96)
        let classifier = AFF_NET_C_O()
        let regressor = AFF_NET_R_O()
        guard let predictedEmotion = try? classifier.prediction(image: pixelBuffer!),
              let predictedValArr = try? regressor.prediction(image: pixelBuffer!) else {
            fatalError("Unexpected runtime error.")
        }
        let valence = predictedValArr.valence_arousal[0].doubleValue
        let arousal = predictedValArr.valence_arousal[1].doubleValue
        // prediction are in [-1,1] so map to [0,1]
        return ((valence+1)/2, (arousal+1)/2, Int(predictedEmotion.emotion)!)
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
        UIApplication.shared.open(URL(string: self.tracks![indexPath.row][2])!, options: [:], completionHandler: nil)
    }
}

extension UIImage {
    public func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        var maybePixelBuffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue]
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         width,
                                         height,
                                         kCVPixelFormatType_32ARGB,
                                         attrs as CFDictionary,
                                         &maybePixelBuffer)
        
        guard status == kCVReturnSuccess, let pixelBuffer = maybePixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer)
        
        guard let context = CGContext(data: pixelData,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
            else {
                return nil
        }
        
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)
        
        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return pixelBuffer
    }
}
