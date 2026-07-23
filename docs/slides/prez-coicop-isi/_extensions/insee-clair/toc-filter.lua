
function Pandoc(doc)
  -- Vérifie si custom-toc est activé
  local custom_toc_enabled = doc.meta and doc.meta["custom-toc"] == true
  if not custom_toc_enabled then return doc end

  -- Récupère toc-title
  local toc_title = ""  -- Valeur par défaut
  if doc.meta and doc.meta["toc-title"] then
    toc_title = pandoc.utils.stringify(doc.meta["toc-title"])
  end

  -- Récupère toc-depth (par défaut: 1)
  local toc_depth = 1
  if doc.meta and doc.meta["toc-depth"] then
    toc_depth = tonumber(doc.meta["toc-depth"]) or 1
  end

  -- Fonction pour lire un fichier
  local function read_file(file)
    local f = io.open(file, "r")
    if not f then error("Fichier introuvable: " .. file) end
    local content = f:read("*a")
    f:close()
    return content
  end

  -- Fonction pour formater les numéros avec des zéros devant
  local function format_number(num)
    return string.format("%02d", num)  -- Ajoute un zéro si < 10
  end

  -- Génère la numérotation et les cartes
  local sections = {}
  local counters = {0}  -- Compteur pour chaque niveau (1-based)
  for i, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.level <= toc_depth then
      -- Vérifie si le titre est marqué comme "unnumbered"
      local is_unnumbered = false
      for _, class in ipairs(block.classes) do
        if class == "unnumbered" then
          is_unnumbered = true
          break
        end
      end

      -- Si le titre est "unnumbered", on l'ignore pour le sommaire
      if is_unnumbered then
        goto continue
      end

      -- Met à jour les compteurs
      while #counters < block.level do
        table.insert(counters, 0)
      end
      counters[block.level] = counters[block.level] + 1
      for j = block.level + 1, #counters do
        counters[j] = 0
      end

      -- Génère le numéro avec zéros devant (ex: "01.02.03")
      local number_parts = {}
      for k = 1, block.level do
        table.insert(number_parts, format_number(counters[k]))
      end
      local section_number = table.concat(number_parts, ".")
      table.insert(sections, {
        id = block.identifier,
        title = pandoc.utils.stringify(block.content),
        level = block.level,
        number = section_number,
        level1_number = format_number(counters[1])  -- Numéro de niveau 1
      })
    end
    ::continue::
  end

  -- Génère les cartes avec classe CSS pour le numéro de niveau 1
  local toc_cards = {}
  for _, section in ipairs(sections) do
    local card_content = string.format(
      '<p class="card-number card-number-%s">%s</p><p></p><p class="card-title">%s</p>',
      section.level1_number, section.number, section.title
    )
    table.insert(toc_cards, string.format(
      '<div class="toc-card level-%d"><a href="#/%s">%s</a></div>',
      section.level, section.id, card_content
    ))
  end

  -- Lit les templates
  local html_template = read_file("_extensions/insee-clair/toc-slide.html")
  local css_content = read_file("_extensions/insee-clair/toc-style.html")
  
  -- Remplace le titre du sommaire dans le template
  html_template = html_template:gsub(
    '<h2>.-</h2>',  -- Cherche n'importe quel titre <h2>...</h2>
    '<h2>' .. toc_title .. '</h2>'  -- Remplace par le titre dynamique
  )

  -- Insère les cartes dans le template
  local final_html = html_template:gsub(
    '<div class="toc%-cards">.-</div>',
    '<div class="toc-cards">' .. table.concat(toc_cards, "\n") .. '</div>'
  )

  -- Ajoute le tout au document
  table.insert(doc.blocks, 1, pandoc.RawBlock('html', final_html .. css_content))
  return doc
end
