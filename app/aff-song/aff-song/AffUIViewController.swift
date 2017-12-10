//
//  AffUIViewController.swift
//  aff-song
//
//  Created by Charlie Hewitt on 10/12/2017.
//  Copyright © 2017 Charlie Hewitt. All rights reserved.
//

import Foundation

class AffUIViewController: UIViewController {
    
    override func viewDidLoad() {
        let gradient = CAGradientLayer()
        gradient.frame = view.bounds
        let c1 = UIColor(red: 252.0/255.0, green: 49.0/255.0, blue: 89.0/255.0, alpha: 1.0)
        let c2 = UIColor(red: 252.0/255.0, green: 45.0/255.0, blue: 119.0/255.0, alpha: 1.0)
        gradient.colors = [c1.cgColor, c2.cgColor]
        view.layer.insertSublayer(gradient, at: 0)
    }
    
}
