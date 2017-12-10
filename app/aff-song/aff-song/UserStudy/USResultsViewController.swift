//
//  USResultsViewController.swift
//  aff-song
//
//  Created by Charlie Hewitt on 10/12/2017.
//  Copyright © 2017 Charlie Hewitt. All rights reserved.
//

import Foundation

class USResultsViewController: ResultsViewController, UINavigationControllerDelegate {
    
    var emotion: Int?
    @IBOutlet weak var StarRatingView: HCSStarRatingView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.identifier == "usAnnotateSegue" {
            // TODO save rating
            print(StarRatingView.value)
            if let nextViewController = segue.destination as? USAnnotateViewController {
                nextViewController.emotion = self.emotion
                nextViewController.faceImage = self.faceImage
            }
        }
    }

}
