//
//  ResultsViewController.swift
//  aff-song
//
//  Created by Charlie Hewitt on 08/12/2017.
//  Copyright © 2017 Charlie Hewitt. All rights reserved.
//

import UIKit
import CoreML

class ResultsViewController: UIViewController, UITableViewDataSource, UITableViewDelegate {
    
    var spotifyAuthKey : String = ""
    var faceImage : UIImage?
    var tracks : [[String]]?
    @IBOutlet var imageView: UIImageView!
    @IBOutlet weak var BackButton: UIButton!
    @IBOutlet weak var TrackTable: UITableView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        let gradient = CAGradientLayer()
        gradient.frame = view.bounds
        let c1 = UIColor(red: 252.0/255.0, green: 49.0/255.0, blue: 89.0/255.0, alpha: 1.0)
        let c2 = UIColor(red: 252.0/255.0, green: 45.0/255.0, blue: 119.0/255.0, alpha: 1.0)
        gradient.colors = [c1.cgColor, c2.cgColor]
        view.layer.insertSublayer(gradient, at: 0)
        BackButton.layer.cornerRadius = BackButton.layer.frame.height/2
        let (valence, arousal) = getAffect(image: faceImage)
        imageView.image = faceImage
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
        let urlPath: String = "https://api.spotify.com/v1/recommendations?seed_genres=pop,rock,dance&valence=\(valence)&danceability=\(arousal)&limit=5&market=GB"
        print(urlPath)
        var request: URLRequest = URLRequest(url: URL(string: urlPath)!)
        request.httpMethod = "GET"
        request.addValue("Bearer " + self.spotifyAuthKey, forHTTPHeaderField: "Authorization")
        
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
