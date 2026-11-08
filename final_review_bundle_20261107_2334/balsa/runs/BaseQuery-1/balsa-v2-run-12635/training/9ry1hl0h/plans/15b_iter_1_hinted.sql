/*+ HashJoin(k mk cn at it1 ct mc mi t)
 MergeJoin(mk cn at it1 ct mc mi t)
 MergeJoin(cn at it1 ct mc mi t)
 MergeJoin(at it1 ct mc mi t)
 MergeJoin(it1 ct mc mi t)
 HashJoin(ct mc mi t)
 HashJoin(mc mi t)
 HashJoin(mi t)
 IndexScan(k)
 SeqScan(mk)
 SeqScan(cn)
 SeqScan(at)
 SeqScan(it1)
 IndexScan(ct)
 SeqScan(mc)
 SeqScan(mi)
 SeqScan(t)
 Leading((k (mk (cn (at (it1 (ct (mc (mi t))))))))) */
SELECT MIN(mi.info) AS release_date,
       MIN(t.title) AS youtube_movie
FROM aka_title AS at,
     company_name AS cn,
     company_type AS ct,
     info_type AS it1,
     keyword AS k,
     movie_companies AS mc,
     movie_info AS mi,
     movie_keyword AS mk,
     title AS t
WHERE cn.country_code = '[us]'
  AND cn.name = 'YouTube'
  AND it1.info = 'release dates'
  AND mc.note LIKE '%(200%)%'
  AND mc.note LIKE '%(worldwide)%'
  AND mi.note LIKE '%internet%'
  AND mi.info LIKE 'USA:% 200%'
  AND t.production_year BETWEEN 2005 AND 2010
  AND t.id = at.movie_id
  AND t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND mk.movie_id = mi.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = at.movie_id
  AND mi.movie_id = mc.movie_id
  AND mi.movie_id = at.movie_id
  AND mc.movie_id = at.movie_id
  AND k.id = mk.keyword_id
  AND it1.id = mi.info_type_id
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id;

