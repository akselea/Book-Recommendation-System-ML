# Proyek Akhir Machine Learning Terapan - *Book Recommendation System*
**Dibuat oleh Aksel E**

Berikut merupakan Proyek Akhir mengenai *Recommendation System* untuk memprediksi buku yang layak dibaca berdasarkan *genre* dari buku tersebut.

## *Project Overview*
<br>
<div><img src="https://images.pexels.com/photos/2908984/pexels-photo-2908984.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"></div>
<br>

Gambar 1. Kumpulan Buku pada Rak Buku

Ditengah kondisi zaman yang semakin berkembang pesat, kebutuhan setiap manusia pun akan semakin bertambah demi memenuhi kebutuhan hariannya. Mudahnya akses untuk mendapatkan apa yang dibutuhkan oleh manusia merupakan salah satu cara agar setiap manusia bisa memenuhi kebutuhannya tanpa mengganggu aktivitas yang ada. Dengan ada nya banyak jenis *online shop* saat ini, setiap orang bisa membeli barang yang dibutuhkan tanpa harus repot datang ke toko dan berinteraksi dengan macetnya jalanan saat ini. Tapi, terkadang kita lupa membeli barang yang kita butuhkan atau barang yang kita inginkan saat memebeli produk barang lain. Disinilah sistem rekomendasi memiliki perannya.

Sistem rekomendasi buku merupakan hasil pengembangan dari teknologi kecerdasan buatan (AI) dan analisis data yang telah berkembang sejak beberapa dekade lalu. Pada awalnya, teknologi ini digunakan untuk merekomendasikan produk pada toko online seperti Amazon dan Netflix. Namun, dengan semakin banyaknya jumlah buku yang tersedia, baik di toko buku maupun perpustakaan, membuat pencarian buku yang cocok dengan preferensi pembaca semakin sulit.

Oleh karena itu, sistem rekomendasi buku dikembangkan untuk membantu pengguna menemukan buku yang tepat berdasarkan preferensi dan minat mereka. Hal ini memungkinkan pembaca untuk mengeksplorasi buku-buku baru yang mungkin belum mereka ketahui, meningkatkan peluang mereka untuk menikmati buku-buku yang mereka baca, serta memberikan pengalaman yang lebih personal dan efisien dalam pencarian buku.

Referensi: [Book Recommendation System through content based and collaborative filtering method](https://ieeexplore.ieee.org/abstract/document/7684166/)

## *Business Understanding*

### *Problem Statements*
Dengan melihat latar belakang tersebut, terdapat beberapa masalah yang didapat:
- Bagaimana melakukan pemrosesan data yang baik agar data dapat digunakan untuk melatih model?
- Fitur apa yang memiliki pengaruh terhadap sistem rekomendasi buku?
- Teknik apa yang lebih baik yang dapat digunakan untuk membuat sistem rekomendasi buku?

### *Goals*
Berikut beberapa solusi yang dapat dilakukan untuk menjawab pertanyaan di atas:
- Melakukan pembersihan data dengan baik hingga dapat digunakan untuk melatih model.
- Mempelajari dan melihat korelasi data agar mendapatkan fitur yang berpengaruh terhadap sistem rekomendasi buku.
- Mencoba teknik yang biasanya digunakan dalam sistem rekomendasi buku.

### *Solution Statements*
- Menggunakan teknik yang kerap digunakan untuk sistem rekomendasi buku (*Content Based Filtering*).

## *Data Understanding*
Data yang digunakan dalam proyek ini merupakan data mengenai buku dengan beberapa data yang berkaitan dengan buku tersebut. Dalam *dataset* ini, terdapat sekitar 7 ribu data dengan 12 kolom didalamnya. Data ini dapat diunduh dari situs Kaggle.

*Link* menuju data: [7k Books](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)

### Berikut merupakan beberapa kolom yang terdapat dalam *dataset* *7k Books*:
- isbn13 : Berupa data ISBN buku yang berkaitan dengan 13 digit angka.
- isbn10 : Berupa data ISBN buku yang berkaitan dengan 10 digit angka.
- title : Merupakan judul dari buku yang berkaitan.
- subtitle : Merupakan subjudul dari buku pada kolom title.
- authors : Merupakan nama penulis dari buku tersebut.
- categories : Merupakan jenis *genre* dari buku tersebut.
- thumbnail : Merupakan gambar *cover* dari buku tersebut pada situs *books.google.com*.
- description : Merupakan deskripsi singkat dari buku tersebut.
- published_year : Berupa tahun dari buku itu dirilis.
- average_rating : Berupa nilai rata-rata *rating* untuk buku tersebut di situs *Goodreads*.
- num_pages : Berupa jumlah halaman pada buku tersebut.
- ratings_count : Berupa jumlah *rating* pada buku tersebut di situs *Goodreads*.

Agar data ini dapat diproses dengan baik oleh model, maka dilakukan beberapa langkah sebagai berikut:

### *Exploratory Data Analysis*
Sebelum nantinya data aka diproses oleh model, langkah yang dilakukan awalnya adalah mencari *missing value* pada setiap kolom data, menghapus fitur yang tidak relevan terhadap sistem rekomendasi buku, serta menentukan data mana yang berpengaruh terhadap sistem rekomendasi buku.

- Menghapus kolom yang tidak memiliki hubungan atau tidak relevan dengan sistem rekomendasi buku, yaitu kolom 'subtitle', 'description', isbn10', 'num_pages', 'ratings_count', dan 'thumbnail' dengan fungsi `.drop()`.
- Menghapus data yang hilang (*missing value*) pada *dataset* dengan fungsi `.dropna()`.

## *Data Preparation*
Dalam bagian ini, ada beberapa hal yang akan dilakukan guna menunjang data yang baik untuk digunakan untuk melatih model. Berikut data yang sudah dibersihkan pada tahap sebelumnya.

Tabel 1. *Dataset 7k Books* Setelah Proses *Exploratory Data Analysis*

|        isbn13 |          title |                         authors |                    categories | published_year | average_rating |
|--------------:|---------------:|--------------------------------:|------------------------------:|---------------:|---------------:|
| 9780002005883 |         Gilead |              Marilynne Robinson |                       Fiction |           2004 |           3.85 |
| 9780002261982 |   Spider's Web | Charles Osborne;Agatha Christie | Detective and mystery stories |           2000 |           3.83 |
| 9780006163831 |   The One Tree |            Stephen R. Donaldson |              American fiction |           1982 |           3.97 |
| 9780006178736 | Rage of angels |                  Sidney Sheldon |                       Fiction |           1993 |           3.93 |
| 9780006280897 | The Four Loves |             Clive Staples Lewis |                Christian life |           2002 |           4.15 |

- **Menghapus Kategori Buku**

  Dalam kolom 'categories', saat dicek menggunakan fungsi `.value_counts()` ternyata kolom ini memiliki lebih dari 560 kategori buku dimana ini merupakan jumlah yang sangat banyak.
  
  <img width="258" alt="image" src="https://user-images.githubusercontent.com/116968275/218927554-32248f98-e848-4cb6-b88e-c2a6da5a1a8f.png">

  Gambar 2. Penyebaran Data pada Kolom *categories*
  
  Melihat terdapat kategori buku yang memiliki jumlah 1 buku saja, maka diputuskan bahwa kategori yang memiliki jumlah buku dibawah 10 akan dihapus dari dataset.

- **Membersihkan Judul Buku**

  Dalam kolom 'title', terdapat beberapa judul buku yang memiliki duplikasi data. Maka dari itu data duplikat ini akan dihapus dari *dataset* guna memudahkan model untuk memberi rekomendasi buku nantinya dengan fungsi `.drop_duplicates()`, serta dilakukannya *cleaning* data pada judul buku ini guna menghapus tanda baca yang terdapat pada judul buku dan menggantinya dengan *whitespace* menggunakan fungsi `re.sub()`.
  
  <img width="234" alt="image" src="https://user-images.githubusercontent.com/116968275/218928482-3cf3a6de-d1ae-474f-9ed6-7e0c366e9899.png">
  
  Gambar 3. Penyebaran Data pada Kolom *title* Sebelum Pembersihan
  
  <img width="232" alt="image" src="https://user-images.githubusercontent.com/116968275/218928577-1814cd03-7f90-47ce-919d-979d7a7218a7.png">

  Gambar 4. Penyebaran Data pada Kolom *title* Setelah Pembersihan

- ***Reset Index* pada *Dataset***

  Setelah beberapa proses dilakukan, proses terakhir adalah me*reset* kembali *index* pada *dataset* agar saat model ingin memanggil data yang dibutuhkan, tidak terjadi *error* akibat nilai *index* melebihi total data yang ada pada *dataset*. Proses ini dilakukan menggunakan fungsi `.reset_index()`.
  
- **Merapihkan Nilai *Rating***

  Setelah proses di atas telah dilakukan, maka proses selanjutnya adalah membuat variabel baru dengan nama 'rating_df' yang akan digunakan untuk melatih model agar dapat merekomendasikan buku kepada pengguna, akan tetapi saat ditampilkan pada variabel 'rating_df' terdapat nilai *rating* buku yang memiliki nilai 0. Maka dari itu, buku yang memiliki nilai *rating* = 0 akan dihapus.
  
  Tabel 2. Data pada 'rating_df' Sebelum Nilai *Rating* Dirapihkan
  
  |                                             title |                      category | rating |
  |--------------------------------------------------:|------------------------------:|-------:|
  |                                            Gilead |                       Fiction |   3.85 |
  |                                      Spider s Web | Detective and mystery stories |   3.83 |
  |                                      The One Tree |              American fiction |   3.97 |
  |                                    Rage of angels |                       Fiction |   3.93 |
  |                                Master of the Game |             Adventure stories |   4.11 |
  |                                               ... |                           ... |    ... |
  |                               Journey to the East |             Adventure stories |   3.70 |
  | The Monk Who Sold His Ferrari A Fable About Fu... |              Health & Fitness |   3.82 |
  |                                         I Am that |                    Philosophy |   4.51 |
  |                          The Berlin Phenomenology |                       History |   0.00 |
  |                           I m Telling You Stories |            Literary Criticism |   3.70 |
  
  Tabel 3. Data pada 'rating_df' Setelah Nilai *Rating* Dirapihkan
  
  |                                             title |                      category | rating |
  |--------------------------------------------------:|------------------------------:|-------:|
  |                                            Gilead |                       Fiction |   3.85 |
  |                                      Spider s Web | Detective and mystery stories |   3.83 |
  |                                      The One Tree |              American fiction |   3.97 |
  |                                    Rage of angels |                       Fiction |   3.93 |
  |                                Master of the Game |             Adventure stories |   4.11 |
  |                                               ... |                           ... |    ... |
  |                              Aspects of the Novel |               English fiction |   3.83 |
  |                               Journey to the East |             Adventure stories |   3.70 |
  | The Monk Who Sold His Ferrari A Fable About Fu... |              Health & Fitness |   3.82 |
  |                                         I Am that |                    Philosophy |   4.51 |
  |                           I m Telling You Stories |            Literary Criticism |   3.70 |
  
## *Modeling and Result*
Pada tahap ini, akan dilakukan pengujian dan pelatihan model menggunakan teknik *Content Based Filtering*.

- ***Content Based Filtering***

  *Content Based Filtering* merupakan salah satu teknik dalam sistem rekomendasi yang menggunakan informasi atau konten dari item yang direkomendasikan (dalam konteks sistem rekomendasi buku, bisa berupa informasi dari buku) untuk membuat rekomendasi.

   Dalam *Content Based Filtering*, setiap item (buku) dijelaskan dengan serangkaian atribut atau fitur, seperti kategori, penulis, tahun terbit, atau kata kunci yang ada dalam sinopsis atau deskripsi buku. Setiap pengguna juga dijelaskan dengan serangkaian preferensi atau minat mereka, berdasarkan atribut atau fitur yang sama dengan yang digunakan untuk deskripsi buku.

  Kemudian, sistem menggunakan algoritma untuk membandingkan atribut atau fitur buku dengan preferensi pengguna untuk menentukan seberapa cocok buku dengan pengguna. Buku-buku yang memiliki atribut atau fitur yang paling mirip dengan preferensi pengguna akan diberikan prioritas lebih tinggi dalam daftar rekomendasi.
  
  Dalam penggunaan teknik *Content Based Filtering* ini, terdapat beberapa fungsi yang digunakan terlebih dahulu:
  - `TfidfVectorizer()` : Berfungsi untuk menghitung bobot kata-kata dalam suatu dokumen. Bobot kata-kata dihitung dengan menggabungkan dua faktor, yaitu frekuensi kata dalam dokumen (*Term Frequency*) dan kepentingan kata tersebut dalam seluruh dokumen (*Inverse Document Frequency*). Dengan fungsi `TfidfVectorizer()`, dokumen teks diubah menjadi vektor numerik, di mana setiap elemen vektor merepresentasikan bobot kata-kata dalam dokumen tersebut. Vektor ini kemudian dapat digunakan sebagai fitur dalam teknik *Content Based Filtering*.

  - `cosine_similarity()` : Digunakan untuk mengukur seberapa mirip antara vektor fitur buku dan preferensi pengguna. Semakin tinggi nilai `cosine_similarity()` antara dua vektor, semakin mirip keduanya. Dalam teknik *Content Based Filtering*, `cosine_similarity()` dapat digunakan untuk menghitung kesesuaian antara vektor fitur buku yang dihasilkan dari `TfidfVectorizer()` dengan preferensi pengguna yang dinyatakan dalam bentuk vektor fitur. Hasil `TfidfVectorizer()` ini dapat digunakan untuk menentukan urutan rekomendasi buku yang paling sesuai dengan preferensi pengguna.

  Dalam penggunaannya, terdapat beberapa kelebihan dan kekurangan dari teknik *Content Based Filtering* ini:
  - **Kelebihan**

    1. Teknik ini cenderung memberikan rekomendasi yang lebih spesifik, karena model hanya berfokus pada fitur yang relevan dengan buku tersebut.
    2. Teknik ini pun dapat menghasilkan rekomendasi yang sesuai, walaupun buku tersebut tidak memiliki riwayat lebih atau nilai *rating* yang tinggi.

  - **Kekurangan**

    1. Rekomendasi yang direkomendasikan cenderung sama seperti buku yang telah dibaca oleh pengguna.

  Dengan penggunaan teknik *Content Based Filtering* ini, diujikan model untuk memberikan 5 rekomendasi buku berdasarkan buku ***The Da Vinci Code***.
  
  Tabel 4. Data Buku ***The Da Vinci Code***
  
  |             title | category | rating |
  |------------------:|---------:|-------:|
  | The Da Vinci Code |  Fiction |   3.82 |
  
  Tabel 5. Hasil Rekomendasi Sistem dengan Teknik *Content Based Filtering* berdasarkan Buku ***The Da Vinci Code***
  
  |                                             title |          category |
  |--------------------------------------------------:|------------------:|
  |                                   Cry the Peacock |           Fiction |
  |                                         I Am that |        Philosophy |
  | The Monk Who Sold His Ferrari A Fable About Fu... |  Health & Fitness |
  |                               Journey to the East | Adventure stories |
  |                              Aspects of the Novel |   English fiction |
 
## *Evaluation*
Pada tahap ini, akan dilakukan evaluasi dari hasil rekomendasi buku pada teknik *Content Based Filtering*.

Untuk melihat relevan atau tidaknya hasil yang sudah model berikan, dapat digunakan rumus untuk mencari nilai *precision* sebagai berikut:

<img width="427" alt="image" src="https://user-images.githubusercontent.com/116968275/218938783-d4949dee-cb35-4df6-9fcc-21f70bc5e9dc.png">

Gambar 5. Rumus Mencari Nilai *Precision*

**Penjelasan**:
- P : Nilai *Precision*
- #*of our recommendations that are relevant* : Jumlah hasil rekomendasi model yang relevan
- #*of items we recommended* : Jumlah rekomendasi model

Dengan menggunakan rumus di atas dan melihat hasil rekomendasi dari model pada Tabel 5, maka didapatkan nilai *Precision*:

Tabel 6. Hasil Rekomendasi Sistem dengan Teknik *Content Based Filtering* berdasarkan Buku ***The Da Vinci Code***
  
|                                             title |          category |           Relevan |
|--------------------------------------------------:|------------------:|------------------:|
|                                   Cry the Peacock |           Fiction |               Iya |
|                                         I Am that |        Philosophy |             Tidak |
| The Monk Who Sold His Ferrari A Fable About Fu... |  Health & Fitness |             Tidak |
|                               Journey to the East | Adventure stories |               Iya |
|                              Aspects of the Novel |   English fiction |               Iya |
 
**P = 3/5 = 0,6 (60%)**
 
Didapatkanlah nilai *Precision* sebesar 60%. Hasil ini masih terbilang kurang baik, dikarenakan referensi yang diberikan tidak memiliki kategori yang sama seperti buku yang menjadi acuan (*Fiction*).
