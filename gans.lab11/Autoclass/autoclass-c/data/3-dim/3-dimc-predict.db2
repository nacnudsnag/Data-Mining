!#; AutoClass C data file -- extension .db2
!#; prior to the first non-comment line being read
!#; the following chars in column 1 make the line a comment:
!#; '!', '#', ';', ' ', and '\n' (empty line)

!#; after the first non-comment line is read, the only column 1 comment characters are
!#; ' ', '\n' (empty line), and comment_char (data file format def in .hd2 file)

;  Two dimensional distribution generated by:
;(let* ((n-data 250)
;       (db (gen-formatted-data `((,n-data ((0.0  9.0) (0.0  3.0) (0.0  1.0)))
;				 (,n-data ((0.0  9.0) (0.0  1.5) (0.0  1.0)))
;				 (,n-data ((0.0  2.0) (0.0  1.0) (0.0  1.0)))
;				 (,n-data ((0.0  0.5) (0.0  0.5) (0.0  1.0))))
;			       ))
;       (total-data (cadr db))
;       (data (map 'vector #'(lambda (x) (coerce x 'vector)) (cddr db))))
;  (Rotate-Data data (/ *single-pi* 6.0) :start 		   0 :end (* 1 n-data))
;  (Rotate-Data data (/ *single-pi* -4.0) :start (* 1 n-data) :end (* 2 n-data))
;  (Shift-Data data #(5.0 3.5 nil) :start   	     0 :end (* 1 n-data))
;  (Shift-Data data #(3.0 4.5 nil) :start (* 1 n-data) :end (* 2 n-data))
;  (Shift-Data data #(8.0 5.0 nil) :start (* 2 n-data) :end (* 3 n-data))
;  (Shift-Data data #(4.0 1.0 nil) :start (* 3 n-data))
  (format nil "; ~A data~2%~A" total-data data))

; 10 Data( 1-based case #'s: 10 20 30 40 50 60 70 80 90 100, 3 attributes

9.134876 6.049179 1.859805 
10.016619 10.84547 -0.9010529 
17.24001 9.96601 0.8749883 
-15.994501 -8.135718 0.80787647 
-3.768199 2.550647 -0.49896044 
-11.673483 -7.674114 -0.23295523 
1.4615703 -0.4979577 -0.7849662 
15.407906 8.627243 -0.74877954 
3.5484169 2.5386896 0.026939131 
1.8351529 1.5959675 -0.7965551 
