
/*Gestion de l'apparition du footer et du nom de la réunion*/

  var $footerPerso = document.getElementById('footerPersonnalise');
  var $pageDebut = document.getElementById('title-slide');
  var $pageFin = document.getElementById('pageDeFin');
  var $toc = document.getElementById("custom-toc-slide");
  var $logoParDefault = document.getElementsByClassName("slide-logo");
  var $header = document.getElementsByClassName("reveal-header");
  
  /*reveal-header*/

  Reveal.on( 'slidechanged', event => {
      
    /*gestion footer*/

    if (!$footerPerso.classList.contains("divCachee")) {
      $footerPerso.classList.add("divCachee");
    } 
    
    if ($pageDebut.classList.contains("present")
        || $pageFin.classList.contains("present")) {
      $footerPerso.classList.remove("divCachee");
    }
    
    /*gestion logo*/
    
    if (!$logoParDefault[0].classList.contains("divCachee")) {
      $logoParDefault[0].classList.add("divCachee");
    }
    
    if ($pageDebut.classList.contains("present")
        || $pageFin.classList.contains("present")) {
      $logoParDefault[0].classList.remove("divCachee");
    }
    
  });


//  modif_number_titre1
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('h1 .header-section-number').forEach(function(el) {
        let num = el.textContent.trim();
        if (num.length === 1) {
            el.textContent = '0' + num;
        }
    });
});


// modif_libellé_titre1
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.reveal h1').forEach(function(h1) {
        // On cherche le numéro de section (supposé être le premier enfant ou dans un .slide-number)
        let numberElement = h1.querySelector('.header-section-number');
        let numberText = numberElement ? numberElement.textContent.trim() : '';
        let fullText = h1.textContent.trim();

        // Si le numéro est présent, on le sépare du reste du titre
        if (numberElement) {
            // On extrait le libellé du titre (après le numéro)
            let titleText = fullText.replace(numberText, '').trim();

            // On vide le h1
            h1.innerHTML = '';

            // On ajoute le numéro (déjà dans un .slide-number)
            h1.appendChild(numberElement);

            // On ajoute un saut de ligne (br) pour séparer visuellement
            h1.appendChild(document.createElement('br'));

            // On enveloppe le libellé dans une span avec la classe title-text
            let titleSpan = document.createElement('span');
            titleSpan.className = 'title-text';
            titleSpan.textContent = titleText;
            h1.appendChild(titleSpan);
        }
    });
});